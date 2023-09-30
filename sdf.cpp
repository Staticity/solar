#include <Eigen/Eigen>
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <Sophus/se3.hpp>
#include <fmt/format.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static void error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error: %s\n", description);
}

std::string readFile(const char* filePath) {
  std::string content;
  std::ifstream fileStream(filePath, std::ios::in);

  if (!fileStream.is_open()) {
    std::cerr << "Could not read file " << filePath << ". File does not exist."
              << std::endl;
    return "";
  }

  std::stringstream sstr;
  sstr << fileStream.rdbuf();
  content = sstr.str();
  fileStream.close();

  return content;
}

bool usePerspective = true;
void keyCallback(
    GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS && key == GLFW_KEY_P) {
    usePerspective = !usePerspective;
  }
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}

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
  SDFSphere(float radius, const Sophus::SE3f& T_sphere_world)
      : radius(radius), T_self_world(T_sphere_world) {}

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

GLuint CreateTexture(
    unsigned char* data, int width, int height, int nrChannels) {
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);

  // set the texture wrapping/filtering options (on the currently bound texture
  // object)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(
      GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  if (data) {
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        (nrChannels == 1) ? GL_RED : GL_RGB,
        width,
        height,
        0,
        (nrChannels == 1) ? GL_RED : GL_RGB,
        GL_UNSIGNED_BYTE,
        data);
    // Cover grayscale case
    if (nrChannels == 1) {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
    }
    glGenerateMipmap(GL_TEXTURE_2D);
  } else {
    return -1;
  }

  return texture;
}

GLuint CreateTexture(unsigned char r, unsigned char g, unsigned char b) {
  std::vector<unsigned char> data = {r, g, b};
  return CreateTexture(data.data(), 1, 1, 3);
}

GLuint CreateTexture(const std::string& filepath) {
  // load and generate the texture
  int width, height, nrChannels;
  unsigned char* data =
      stbi_load(filepath.c_str(), &width, &height, &nrChannels, 0);

  const auto texture = CreateTexture(data, width, height, nrChannels);
  stbi_image_free(data);
  return texture;
}

GLuint CreateShaderProgram() {
  const std::string vertexShaderSource =
      readFile("/Users/static/Documents/code/sdfs/shaders/sdf.vert");
  const std::string fragmentShaderSource =
      readFile("/Users/static/Documents/code/sdfs/shaders/sdf.frag");
  const char* pVertexShaderSource = vertexShaderSource.c_str();
  const char* pFragmentShaderSource = fragmentShaderSource.c_str();

  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &pVertexShaderSource, NULL);
  glCompileShader(vertexShader);

  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &pFragmentShaderSource, NULL);
  glCompileShader(fragmentShader);

  GLuint shaderProgram = glCreateProgram();

  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  return shaderProgram;
}

struct Lighting {
  bool isMatte;
};

struct TexturedObject {
  TexturedObject(
      std::shared_ptr<SDF> object,
      GLuint texture,
      const Lighting& lighting = {.isMatte = false})
      : parent(nullptr), object(object), texture(texture), lighting(lighting) {
    std::vector<SDFType> sdfTypes;
    std::tie(sdfTypes, indices, params) =
        SDFTree::CreateTreeFromPrefix({object}).serialize();

    for (const auto x : sdfTypes) {
      types.push_back(static_cast<int>(x));
    }
    while (params.size() % 4 != 0) {
      params.push_back(0);
    }
  }

  void setParent(TexturedObject* parent) { this->parent = parent; }

  TexturedObject* parent;

  std::shared_ptr<SDF> object;
  GLuint texture;

  std::vector<int> types;
  std::vector<int> indices;
  std::vector<float> params;

  Lighting lighting;
};

int main() {
  glfwSetErrorCallback(error_callback);

  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window =
      glfwCreateWindow(800, 600, "Raymarching SDF", NULL, NULL);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);

  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    return -1;
  }

  const GLubyte* version = glGetString(GL_VERSION);
  printf("OpenGL Version: %s\n", version);

  float vertices[] = {
      -1.0f,
      1.0f,
      -1.0f,
      -1.0f,
      1.0f,
      -1.0f,

      1.0f,
      -1.0f,
      1.0f,
      1.0f,
      -1.0f,
      1.0f};

  GLuint VAO, VBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);

  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  const auto shaderProgram = CreateShaderProgram();
  glUseProgram(shaderProgram);
  GLint isMatteLoc = glGetUniformLocation(shaderProgram, "isMatte");
  GLint numElementsLoc = glGetUniformLocation(shaderProgram, "numElements");
  GLint intrinsicsLoc = glGetUniformLocation(shaderProgram, "K");
  GLint T_world_cameraLoc =
      glGetUniformLocation(shaderProgram, "T_world_camera");
  GLint light_worldLoc = glGetUniformLocation(shaderProgram, "light_world");
  GLint treeShapeTypesLoc =
      glGetUniformLocation(shaderProgram, "treeShapeTypes");
  GLint treeParametersIndexLoc =
      glGetUniformLocation(shaderProgram, "treeParametersIndex");
  GLint shapeParametersLoc =
      glGetUniformLocation(shaderProgram, "shapeParameters");

  auto start = std::chrono::high_resolution_clock::now();
  const auto sunTexture = // CreateTexture(255, 0, 0);
      CreateTexture("/Users/static/Documents/code/sdfs/build/assets/sun.jpg");
  const auto moonTexture = // CreateTexture(0, 255, 0);
      CreateTexture("/Users/static/Documents/code/sdfs/build/assets/moon.jpg");
  const auto earthTexture = // CreateTexture(0, 0, 255);
      CreateTexture("/Users/static/Documents/code/sdfs/build/assets/earth.jpg");
  const auto end = std::chrono::high_resolution_clock::now();
  fmt::println("Loaded textures in {} seconds", (end - start).count() / 1e9);

  const auto sunIndex = 0;
  std::vector<TexturedObject> spheres = {
      {std::make_shared<SDFSphere>(
           1400, Sophus::SE3f(Sophus::SO3f(), Eigen::Vector3f{-150000, 0, 0})),
       sunTexture,
       Lighting{.isMatte = true}},
      {std::make_shared<SDFSphere>(
           12.576,
           //  .1,
           Sophus::SE3f(
               Sophus::SO3f::rotX(-M_PI / 2.0) * Sophus::SO3f::rotY(M_PI / 2),
               //  Sophus::SO3f::rotZ(22.5 * M_PI / 180.0),
               Eigen::Vector3f{0, 0, 0})),
       earthTexture,
       Lighting{.isMatte = false}},
      {std::make_shared<SDFSphere>(
           3.475, Sophus::SE3f(Sophus::SO3f(), Eigen::Vector3f{384.4, 0, 0})),
       moonTexture,
       Lighting{.isMatte = false}} //,
      // {std::make_shared<SDFCylinder>(
      //      .1,
      //      15,
      //      Sophus::SE3f(
      //          Sophus::SO3f::rotZ(22.5 * M_PI / 180.0),
      //          Eigen::Vector3f::Zero())),
      //  CreateTexture(255, 0, 0),
      //  Lighting{.isMatte = true}},
      // {std::make_shared<SDFCylinder>(
      //      1,
      //      15,
      //      Sophus::SE3f(Sophus::SO3f::rotX(M_PI / 2),
      //      Eigen::Vector3f::Zero())),
      //  CreateTexture(0, 255, 0),
      //  Lighting{.isMatte = true}},
      // {std::make_shared<SDFCylinder>(
      //      1,
      //      15,
      //      Sophus::SE3f(Sophus::SO3f::rotZ(M_PI / 2),
      //      Eigen::Vector3f::Zero())),
      //  CreateTexture(0, 0, 255),
      //  Lighting{.isMatte = true}}
  };

  start = std::chrono::high_resolution_clock::now();
  glEnable(GL_DEPTH_TEST);
  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set intrinsics matrix
    // clang-format off
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    const auto f = std::min(width, height);
    Eigen::Matrix3f K;
    K << 
      f / 2.0,       0,  width / 2.0,
            0, f / 2.0, height / 2.0,
            0,       0,            1;
    // clang-format on

    // Calculate current time
    const auto timeSecs =
        (start - std::chrono::high_resolution_clock::now()).count() / 1e9;

    // Set camera pose
    Sophus::SE3f T_camera_world(
        Sophus::SO3f::rotY(timeSecs * M_PI / 4), Eigen::Vector3f{0, 0, 30});
    const auto matT_world_camera = T_camera_world.inverse().matrix();

    const Eigen::Vector3f light_world = Eigen::Vector3f{1, 0, 0};
    // -spheres[0].sphere.T_self_world.inverse().translation().normalized();

    glUseProgram(shaderProgram);
    for (const auto& sphere : spheres) {
      glUniformMatrix3fv(intrinsicsLoc, 1, GL_FALSE, K.data());
      glUniformMatrix4fv(
          T_world_cameraLoc, 1, GL_FALSE, matT_world_camera.data());
      glUniform3fv(light_worldLoc, 1, light_world.data());
      glUniform1i(isMatteLoc, sphere.lighting.isMatte ? 1 : 0);

      glUniform1i(numElementsLoc, sphere.types.size());
      glUniform1iv(treeShapeTypesLoc, sphere.types.size(), sphere.types.data());
      glUniform1iv(
          treeParametersIndexLoc, sphere.indices.size(), sphere.indices.data());
      glUniform4fv(
          shapeParametersLoc, sphere.params.size() / 4, sphere.params.data());

      glBindTexture(GL_TEXTURE_2D, sphere.texture);

      // Draw full-screen quad
      glBindVertexArray(VAO);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glBindVertexArray(0);
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteProgram(shaderProgram);

  glfwTerminate();
  return 0;
}
