#include <Eigen/Eigen>
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <Sophus/se3.hpp>
#include <fmt/format.h>

#include <fstream>
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

  unsigned int texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  // set the texture wrapping/filtering options (on the currently bound texture
  // object)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(
      GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // load and generate the texture
  int width, height, nrChannels;
  unsigned char* data = stbi_load(
      "/Users/static/Documents/code/sdfs/build/assets/moon.jpg",
      &width,
      &height,
      &nrChannels,
      0);
  fmt::println("w: {}, h: {}, ch: {}", width, height, nrChannels);
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
    if (nrChannels == 1) {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
    }
    glGenerateMipmap(GL_TEXTURE_2D);
  } else {
    std::cout << "Failed to load texture" << std::endl;
  }
  stbi_image_free(data);

  const GLubyte* version = glGetString(GL_VERSION);
  printf("OpenGL Version: %s\n", version);

  // Earlier in your code
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

  glUseProgram(shaderProgram);
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

  fmt::println(
      "{} {} {} {} {} {} {}",
      numElementsLoc,
      intrinsicsLoc,
      T_world_cameraLoc,
      light_worldLoc,
      treeShapeTypesLoc,
      treeParametersIndexLoc,
      shapeParametersLoc);

  const auto trees =
      SDFTree::CreateTreesFromPrefix({std::make_shared<SDFSphere>(
          1,
          Sophus::SE3f(
              Sophus::SO3f::rotY(M_PI / 4) * Sophus::SO3f::rotX(M_PI / 2),
              Eigen::Vector3f{0, 0, 0})
              .inverse())});
  auto [treeShapeTypesEnum, treeParametersIndex, shapeParameters] =
      SDFTree::SerializeTrees(trees);
  // multiple of 4 padding
  while (shapeParameters.size() % 4 != 0) {
    shapeParameters.push_back(0);
  }
  // cast to int
  std::vector<int> treeShapeTypes;
  for (const auto x : treeShapeTypesEnum) {
    treeShapeTypes.push_back(static_cast<int>(x));
  }

  const auto start = std::chrono::high_resolution_clock::now();
  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // clang-format off
    Eigen::Matrix3f K;
    K << 
      width / 2,          0,  width / 2,
              0, width / 2, height / 2,
              0,          0,          1;
    // clang-format on

    const auto timeSecs =
        (start - std::chrono::high_resolution_clock::now()).count() / 1e9;

    Sophus::SE3f T_camera_world(
        Sophus::SO3f::rotY(0 * timeSecs * M_PI / 4), Eigen::Vector3f{0, 0, 2});
    const auto matT_world_camera = T_camera_world.inverse().matrix();

    const Eigen::Vector3f light_world =
        Sophus::SO3f::rotY(timeSecs * M_PI / 4) *
        Eigen::Vector3f{1, 0, 0}.normalized();

    // Set uniforms
    glUniform1i(numElementsLoc, treeShapeTypes.size());
    glUniformMatrix3fv(intrinsicsLoc, 1, GL_FALSE, K.data());
    glUniformMatrix4fv(
        T_world_cameraLoc, 1, GL_FALSE, matT_world_camera.data());
    glUniform3fv(light_worldLoc, 1, light_world.data());
    glUniform1iv(
        treeShapeTypesLoc, treeShapeTypes.size(), treeShapeTypes.data());
    glUniform1iv(
        treeParametersIndexLoc,
        treeParametersIndex.size(),
        treeParametersIndex.data());
    glUniform4fv(
        shapeParametersLoc, shapeParameters.size() / 4, shapeParameters.data());

    glBindTexture(GL_TEXTURE_2D, texture);

    // Draw full-screen quad
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteProgram(shaderProgram);

  glfwTerminate();
  return 0;
}
