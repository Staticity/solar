#include <Eigen/Eigen>
#include <SFML/Graphics.hpp>
#include <fmt/core.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <chrono>
#include <iostream>
#include <optional>

struct SDFHit {
  int treeIndex;
  Eigen::Vector3f position;
  Eigen::Vector3f normal;
};

enum SDFType : int {
  Complement = -4,
  Union,
  Intersection,
  Subtraction,
  AddRadius,
  Sphere,
  Cylinder,
  None
};

std::string toString(SDFType type) {
  switch (type) {
    case Complement:
      return "Complement";
    case Union:
      return "Union";
    case Intersection:
      return "Intersection";
    case Subtraction:
      return "Subtraction";
    case AddRadius:
      return "AddRadius";
    case Sphere:
      return "Sphere";
    case Cylinder:
      return "Cylinder";
    case None:
      return "None";
    default:
      return "UNKNOWN";
  }
}

struct SDF {
  virtual int numOperands() const { return 0; }
  virtual SDFType type() const { return SDFType::None; }
  virtual int numParams() const { return 0; }
  virtual void storeParams(float* buffer) const {}
};

struct SDFComplement : public SDF {
  virtual int numOperands() const override { return 1; }
  virtual SDFType type() const override { return SDFType::Complement; }
};
struct SDFUnion : public SDF {
  virtual int numOperands() const override { return 2; }
  virtual SDFType type() const override { return SDFType::Union; }
};
struct SDFIntersection : public SDF {
  virtual int numOperands() const override { return 2; }
  virtual SDFType type() const override { return SDFType::Intersection; }
};
struct SDFSubtraction : public SDF {
  virtual int numOperands() const override { return 2; }
  virtual SDFType type() const override { return SDFType::Subtraction; }
};
struct SDFAddRadius : public SDF {
  float radius = 0.0f;

  virtual SDFType type() const override { return SDFType::AddRadius; }
  virtual int numOperands() const override { return 0; }
  virtual int numParams() const override { return 1; }
  virtual void storeParams(float* buffer) const override { buffer[0] = radius; }
};
struct SDFSphere : public SDF {
  SDFSphere(
      float radius,
      const Sophus::SE3f& T_sphere_world = {},
      sf::Image* texture = nullptr)
      : radius(radius), T_self_world(T_sphere_world), texture(texture) {}

  virtual SDFType type() const override { return SDFType::Sphere; }
  virtual int numOperands() const override { return 0; }
  virtual int numParams() const override { return 16 + 1; }
  virtual void storeParams(float* buffer) const override {
    const Eigen::Matrix4f M = T_self_world.matrix();
    std::copy(M.data(), M.data() + 16, buffer);
    buffer[16] = radius;
  }

  Sophus::SE3f T_self_world;
  float radius;
  sf::Image* texture;
};
struct SDFCylinder : public SDF {
  SDFCylinder(float radius, float height, const Sophus::SE3f& T_self_world)
      : radius(radius), height(height), T_self_world(T_self_world) {}

  virtual SDFType type() const override { return SDFType::Cylinder; }
  virtual int numOperands() const override { return 0; }
  virtual int numParams() const override { return 16 + 2; }
  virtual void storeParams(float* buffer) const override {
    const Eigen::Matrix4f M = T_self_world.matrix();
    std::copy(M.data(), M.data() + 16, buffer);
    buffer[16] = radius;
    buffer[17] = height;
  }

  Sophus::SE3f T_self_world;
  float radius;
  float height;
  sf::Image* texture;
};

class SDFTree {
 public:
  void print() const { _print(); }

  int size() const {
    int s = 1;
    for (const auto& child : children) {
      s += child.size();
    }
    return s;
  }

  int numParams() const {
    int num = object->numParams();
    for (const auto& child : children) {
      num += child.numParams();
    }
    return num;
  }

  std::tuple<std::vector<SDFType>, std::vector<int>, std::vector<float>>
  serialize() const {
    std::vector<SDFType> types;
    std::vector<int> indices;
    std::vector<float> params;

    types.resize(this->size());
    indices.resize(types.size());
    params.resize(this->numParams());

    this->_storeTree(0, types.data(), indices.data(), params.data());

    return {types, indices, params};
  }

  static std::tuple<std::vector<SDFType>, std::vector<int>, std::vector<float>>
  SerializeTrees(const std::vector<SDFTree>& trees) {
    std::vector<SDFType> types;
    std::vector<int> indices;
    std::vector<float> params;

    for (const auto& tree : trees) {
      const auto indexOffset = params.size();
      const auto [treeTypes, treeIndices, treeParams] = tree.serialize();
      for (const auto x : treeTypes)
        types.push_back(x);
      for (const auto x : treeIndices)
        indices.push_back(indexOffset + x);
      for (const auto x : treeParams)
        params.push_back(x);
    }

    return {types, indices, params};
  }

  static SDFTree CreateTreeFromPrefix(
      const std::vector<std::shared_ptr<SDF>>& prefix) {
    return std::get<0>(_CreateTreeFromPrefix(prefix));
  }

  static std::vector<SDFTree> CreateTreesFromPrefix(
      const std::vector<std::shared_ptr<SDF>>& prefix) {
    std::vector<SDFTree> trees;
    int i = 0;
    while (i < prefix.size()) {
      const auto [tree, size] = _CreateTreeFromPrefix(prefix, i);
      trees.push_back(tree);
      i += size;
    }
    return trees;
  }

 private:
  void _print(int depth = 0) const {
    for (int i = 0; i < depth; ++i) {
      std::cout << '\t';
    }
    std::cout << toString(object->type()) << std::endl;
    for (const auto& child : children) {
      child._print(depth + 1);
    }
  }

  static std::tuple<SDFTree, int> _CreateTreeFromPrefix(
      const std::vector<std::shared_ptr<SDF>>& prefix, int start = 0) {
    SDFTree tree;
    tree.object = prefix.at(start);

    int size = 1;
    for (int i = 0; i < tree.object->numOperands(); ++i) {
      const auto [child, childSize] =
          _CreateTreeFromPrefix(prefix, start + size);
      tree.children.push_back(child);
      size += childSize;
    }
    return {tree, size};
  }

  std::tuple<int, SDFType*, int*, float*> _storeTree(
      int index,
      SDFType* typeBuffer,
      int* indexBuffer,
      float* paramBuffer) const {
    // Store the current object
    *(typeBuffer++) = object->type();
    *(indexBuffer++) = index;

    object->storeParams(paramBuffer);
    paramBuffer += object->numParams();
    index += object->numParams();

    // Store the parameters of every child
    for (const auto& child : children) {
      std::tie(index, typeBuffer, indexBuffer, paramBuffer) =
          child._storeTree(index, typeBuffer, indexBuffer, paramBuffer);
    }

    // Return the updated buffer positions
    return {index, typeBuffer, indexBuffer, paramBuffer};
  }

  std::shared_ptr<SDF> object;
  std::vector<SDFTree> children;
};

std::pair<int, float> signedDistance(
    const Eigen::Vector3f& position,
    const int treeIndex,
    const std::vector<SDFType>& treeShapeTypes,
    const std::vector<int>& treeParametersIndex,
    const std::vector<float>& shapeParameters) {
  const auto parametersIndex = treeParametersIndex.at(treeIndex);

  const auto shapeType = treeShapeTypes.at(treeIndex);
  if (shapeType < 0) {
    if (shapeType == SDFType::Complement) {
      const auto [size, distance] = signedDistance(
          position,
          treeIndex + 1,
          treeShapeTypes,
          treeParametersIndex,
          shapeParameters);

      return {1 + size, -distance};
    } else if (shapeType == SDFType::AddRadius) {
      const auto [size, distance] = signedDistance(
          position,
          treeIndex + 1,
          treeShapeTypes,
          treeParametersIndex,
          shapeParameters);
      const auto radius = shapeParameters.at(parametersIndex);

      return {1 + size, distance - radius};
    }

    const auto [leftSize, leftDistance] = signedDistance(
        position,
        treeIndex + 1,
        treeShapeTypes,
        treeParametersIndex,
        shapeParameters);
    const auto [rightSize, rightDistance] = signedDistance(
        position,
        treeIndex + leftSize + 1,
        treeShapeTypes,
        treeParametersIndex,
        shapeParameters);

    if (shapeType == SDFType::Union) {
      return {1 + leftSize + rightSize, std::min(leftDistance, rightDistance)};
    } else if (shapeType == SDFType::Intersection) {
      return {1 + leftSize + rightSize, std::max(leftDistance, rightDistance)};
    } else if (shapeType == SDFType::Subtraction) {
      return {1 + leftSize + rightSize, std::max(leftDistance, -rightDistance)};
    } else {
      std::cout << fmt::format("unidentified shape: {}", int(shapeType))
                << std::endl;
      assert(false);
    }
  }

  if (shapeType == SDFType::Sphere) {
    const Eigen::Matrix4f T_sphere_world = Eigen::Map<const Eigen::Matrix4f>(
        shapeParameters.data() + parametersIndex);
    const auto radius = shapeParameters.at(parametersIndex + 16);
    const Eigen::Vector3f position_sphere =
        (T_sphere_world * position.homogeneous()).hnormalized();

    return {1, position_sphere.norm() - radius};
  } else if (shapeType == SDFType::Cylinder) {
    const Eigen::Matrix4f T_sphere_world = Eigen::Map<const Eigen::Matrix4f>(
        shapeParameters.data() + parametersIndex);
    const auto radius = shapeParameters.at(parametersIndex + 16);
    const auto height = shapeParameters.at(parametersIndex + 17);
    const Eigen::Vector3f deltaPosition =
        (T_sphere_world * position.homogeneous()).hnormalized();

    return {
        1,
        // Intersection of:
        // - infinitely tall (vertical) cylinder
        // - infinitely large (horizontal) rectangular prism
        std::max(
            std::hypot(deltaPosition.x(), deltaPosition.z()) - radius,
            std::abs(deltaPosition.y()) - height)};
  }

  return {-1, std::numeric_limits<float>::infinity()};
}

std::optional<SDFHit> raymarch(
    const Eigen::Vector3f& startPosition,
    const Eigen::Vector3f& direction,
    const std::vector<SDFType>& treeShapeTypes,
    const std::vector<int>& treeParametersIndex,
    const std::vector<float>& shapeParameters) {
  // To prevent taking very tiny steps close to objects, let's set a minimum
  // step size. This will make all objects this much thicker.
  constexpr auto MinimumStepSize = 1e-3;

  // This is the maximum distance we can see, to prevent infinite searching.
  constexpr auto MaximumDistance = 1e3f;

  double distanceTravelled = 0.0;
  Eigen::Vector3f position = startPosition;

  while (distanceTravelled < MaximumDistance) {
    std::pair<float, int> minDistanceObject = {
        std::numeric_limits<float>::infinity(), -1};

    // Compute the distance to the nearest object
    int treeIndex = 0;
    while (treeIndex < treeShapeTypes.size()) {
      const auto [size, distance] = signedDistance(
          position,
          treeIndex,
          treeShapeTypes,
          treeParametersIndex,
          shapeParameters);

      minDistanceObject = std::min(minDistanceObject, {distance, treeIndex});
      treeIndex += size;
    }

    // Did we hit the object?
    const auto [minDistance, closestObjectTreeIndex] = minDistanceObject;
    if (minDistance < MinimumStepSize) {
      SDFHit hit{.treeIndex = closestObjectTreeIndex, .position = position};

      // Estimate the normal of the surface. This is to be used for Jacobian
      // (normal) calculations.
      constexpr auto epsilon = 1e-5;
      for (int i = 0; i < 3; ++i) {
        Eigen::Vector3f forward = position;
        Eigen::Vector3f backward = position;
        forward(i) += epsilon;
        backward(i) -= epsilon;

        const auto [_f, forwardDistance] = signedDistance(
            forward,
            closestObjectTreeIndex,
            treeShapeTypes,
            treeParametersIndex,
            shapeParameters);
        const auto [_b, backwardDistance] = signedDistance(
            backward,
            closestObjectTreeIndex,
            treeShapeTypes,
            treeParametersIndex,
            shapeParameters);

        // Limit as epislon approaches 0:
        //     f(x + e) - f(x - e)
        //    ---------------------
        //             2e
        hit.normal(i) = (forwardDistance - backwardDistance) / (2 * epsilon);
      }
      hit.normal.normalize();

      return hit;
    }

    // If not, march the ray forward
    position += direction * minDistance;
    distanceTravelled += minDistance;
  }

  // We never hit the object, return an empty hit.
  return {};
}

// bool validateTree(
//     const std::vector<int>& treeShapeTypes,
//     const std::vector<int>& treeParametersIndex,
//     const std::vector<float>& shapeParameters,
//     int i = 0) {}

sf::Image renderSDFs(
    int width,
    int height,
    const Eigen::Matrix3f& K,
    float angle,
    const sf::Image& earth) {
  sf::Image image;
  image.create(width, height);

  // clang-format off
  const auto trees = SDFTree::CreateTreesFromPrefix({
    std::make_shared<SDFSphere>(
      1,
      Sophus::SE3f(Sophus::SO3f::rotY(M_PI/4) * Sophus::SO3f::rotX(M_PI / 2), Eigen::Vector3f{0, 0, 2}).inverse()
    )
    // ,
    // std::make_shared<SDFSphere>(
    //   1,
    //   Sophus::SE3f( Sophus::SO3f::rotX(-M_PI/2), Eigen::Vector3f{-1, 0, 3}).inverse()
    // ),
  });
  const auto [treeShapeTypes, treeParametersIndex, shapeParameters] = SDFTree::SerializeTrees(trees);

  for (const auto& tree : trees) {
    tree.print();
  }

  const auto [treeShapeTypes, treeParametersIndex, shapeParameters] = SDFTree::SerializeTrees(trees);
  fmt::println("treeShapeTypes");
  for (auto x : treeShapeTypes) fmt::print("{} ", static_cast<int>(x));
  fmt::println("");
  fmt::println("treeParametersIndex");
  for (auto x : treeParametersIndex) fmt::print("{} ", x);
  fmt::println("");
  fmt::println("shapeParameters");
  for (auto x : shapeParameters) fmt::print("{} ", x);
  fmt::println("\n");

  fmt::println("Image size: {} {}", earth.getSize().x, earth.getSize().y);

#if 0
  const std::vector<int> treeShapeTypes = {
      // SDFType::Intersection,
      // SDFType::Sphere,
      // SDFType::Sphere,
      // SDFType::AddRadius,
      SDFType::Sphere
    };
  const std::vector<int> treeParametersIndex = {
    // -1,
    // 0,
    // 4,
    // 0,
    1,
  };
  const std::vector<float> shapeParameters = {
    // 0, 0, 2, 1,
    // 0, 0, 3, .5,
    1,
    0, 0, 2, 1, 1
  };
#endif
  // clang-format on

  int hits = 0;
  const Eigen::Matrix3f Kinv = K.inverse();
  for (int u = 0; u < width; ++u) {
    for (int v = 0; v < height; ++v) {
      const Eigen::Vector3f ray =
          (Kinv * Eigen::Vector3f{u, v, 1}).normalized();

      const auto maybeHit = raymarch(
          Eigen::Vector3f::Zero(),
          ray,
          treeShapeTypes,
          treeParametersIndex,
          shapeParameters);

      if (const auto hit = maybeHit) {
        const Eigen::Vector3f light{std::cos(angle), 0, std::sin(angle)};
        auto angle = std::max(0.0f, light.dot(hit->normal));
        auto d = static_cast<std::uint8_t>(angle * 255);

        const auto objectIndex = hit->treeIndex;
        const auto objectType = treeShapeTypes.at(objectIndex);
        const auto parametersIndex = treeParametersIndex.at(objectIndex);
        if (objectType == SDFType::Sphere) {
          const Eigen::Matrix4f T_sphere_world =
              Eigen::Map<const Eigen::Matrix4f>(
                  shapeParameters.data() + parametersIndex);

          const auto radius = shapeParameters.at(parametersIndex + 16);

          const Eigen::Vector3f position_sphere =
              (T_sphere_world * hit->position.homogeneous()).hnormalized();
          const Eigen::Vector3f direction_sphere = position_sphere.normalized();

          const auto a =
              (std::atan2(direction_sphere.y(), direction_sphere.x()) + M_PI) /
              (2 * M_PI);
          const auto b = std::acos(direction_sphere.z()) / M_PI;
          const auto textureU = int(a * earth.getSize().x);
          const auto textureV = int(b * earth.getSize().y);

          const auto matteColor = earth.getPixel(
              std::clamp(textureU, 0, int(earth.getSize().x - 1)),
              std::clamp(textureV, 0, int(earth.getSize().y - 1)));
          const auto color = sf::Color(
              int(matteColor.r * angle),
              int(matteColor.g * angle),
              int(matteColor.b * angle));
          image.setPixel(u, height - v - 1, color);
        } else if (objectType == SDFType::Cylinder) {
          const Eigen::Vector3f shapePosition{
              shapeParameters.at(parametersIndex),
              shapeParameters.at(parametersIndex + 1),
              shapeParameters.at(parametersIndex + 2)};
          const auto radius = shapeParameters.at(parametersIndex + 3);
          const auto height = shapeParameters.at(parametersIndex + 4);

          const Eigen::Vector3f position_cylinder =
              hit->position - shapePosition;

          const auto a =
              std::hypot(position_cylinder.x(), position_cylinder.z()) / radius;
          const auto b =
              std::atan2(position_cylinder.z(), position_cylinder.x());

          const auto textureU = int(a * earth.getSize().x);
          const auto textureV = int(b * earth.getSize().y);
          const auto matteColor = earth.getPixel(
              std::clamp(textureU, 0, int(earth.getSize().x - 1)),
              std::clamp(textureV, 0, int(earth.getSize().y - 1)));
          angle = 1;
          const auto color = sf::Color(
              int(matteColor.r * angle),
              int(matteColor.g * angle),
              int(matteColor.b * angle));
          image.setPixel(u, height - v - 1, color);
        } else {
          image.setPixel(u, height - v - 1, {d, d, d});
        }

        ++hits;
      } else {
        image.setPixel(u, height - v - 1, {0, 0, 0});
      }
    }
  }

  fmt::println(
      "{:.2f}% of pixels hit an object", hits * 100.0 / (width * height));

  return image;
}

sf::Image createGradientImage(int width, int height) {
  sf::Image image;
  image.create(width, height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      sf::Color color(255 * x / width, 255 * y / height, 0);
      image.setPixel(x, y, color);
    }
  }

  return image;
}

int main() {
  const int width = 512;
  const int height = 512;

  sf::RenderWindow window(sf::VideoMode(width, height), "SFML Demo");

  sf::Image earth;
  earth.loadFromFile("/Users/static/Documents/code/sdfs/build/assets/moon.jpg");

  // clang-format off
  Eigen::Matrix3f K;
  K << width/2, 0, width / 2,
       0, height/2, height / 2,
       0, 0, 1;
  // clang-format on

  constexpr auto AngleStep = 10.0f;
  float angle = -20;
  sf::Texture texture;
  const auto start = std::chrono::high_resolution_clock::now();
  texture.loadFromImage(
      renderSDFs(width, height, K, angle / 180 * M_PI, earth));
  const auto end = std::chrono::high_resolution_clock::now();
  fmt::println("{} seconds", (end - start).count() / 1e9);
  sf::Sprite sprite(texture);

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();
      if (event.type == sf::Event::KeyPressed) {
        if (event.key.code == sf::Keyboard::Left) {
          angle += AngleStep;
          texture.loadFromImage(
              renderSDFs(width, height, K, angle / 180 * M_PI, earth));
        } else if (event.key.code == sf::Keyboard::Right) {
          angle -= AngleStep;
          texture.loadFromImage(
              renderSDFs(width, height, K, angle / 180 * M_PI, earth));
        }
      }
    }

    window.clear();
    window.draw(sprite);
    window.display();
  }

  return 0;
}