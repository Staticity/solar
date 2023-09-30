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
#include "SpiceUsr.h"

#include <sophus/se3.hpp>
#include "includes/aaplus/AA+.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class SolarSystemPose {
 public:
  SolarSystemPose() { init(); }
  ~SolarSystemPose() { cleanup(); }

  Sophus::SE3d T_J2000_body(const std::string& body, double ephemerisTime) {
    SpiceDouble bodyState[6], lt;
    spkezr_c(
        body.c_str(), ephemerisTime, "J2000", "NONE", "EARTH", bodyState, &lt);

    const std::string& fixedFrame = "IAU_" + body;
    SpiceDouble R_J2000_body[3][3];
    pxform_c(fixedFrame.c_str(), "J2000", ephemerisTime, R_J2000_body);
    SpiceDouble q_J2000_body[4];
    m2q_c(R_J2000_body, q_J2000_body);

    return Sophus::SE3d(
        Eigen::Quaterniond(
            q_J2000_body[0], q_J2000_body[1], q_J2000_body[2], q_J2000_body[3]),
        Eigen::Vector3d(bodyState[0], bodyState[1], bodyState[2]) / 1e3);
  }

 private:
  const std::vector<std::string> kernelPaths_ = {
      "/Users/static/Downloads/cspice/data/naif0012.tls",
      "/Users/static/Downloads/cspice/data/de430.bsp",
      "/Users/static/Downloads/cspice/data/pck00011.tpc"};

  void init() {
    for (const auto& path : kernelPaths_) {
      furnsh_c(path.c_str());
    }
  }

  void cleanup() {
    for (const auto& path : kernelPaths_) {
      unload_c(path.c_str());
    }
  }
};

static void error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error: %s\n", description);
}

void keyCallback(
    GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}

Eigen::Vector3f MoonEclipticRectangularCoordinatesJ2000(double JD) noexcept {
  double Longitude{CAAMoon::EclipticLongitude(JD)};
  Longitude = CAACoordinateTransformation::DegreesToRadians(Longitude);
  double Latitude{CAAMoon::EclipticLatitude(JD)};
  Latitude = CAACoordinateTransformation::DegreesToRadians(Latitude);
  const double coslatitude{cos(Latitude)};
  const double R{CAAMoon::RadiusVector(JD)};

  Eigen::Vector3f value;
  value.x() = R * coslatitude * cos(Longitude);
  value.y() = R * coslatitude * sin(Longitude);
  value.z() = R * sin(Latitude);
  return value;
}

Eigen::Vector3f SphericalToCartesian(
    double longitude, double latitude, double radius) {
  Eigen::Vector3d result;
  double cosLat = cos(latitude);
  result.x() = radius * cosLat * cos(longitude);
  result.y() = radius * cosLat * sin(longitude);
  result.z() = radius * sin(latitude);
  return result.cast<float>();
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

GLuint CreateShaderProgram(
    const std::string& vertexShaderSource,
    const std::string& fragmentShaderSource) {
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

struct Camera {
  Sophus::SE3f T_world_self;
  Eigen::Matrix3f K;
};

struct Light {
  Sophus::SE3f T_self_world;
  //   Eigen::Vector3f color;
};

struct SDFObject {
  // 1: Sphere, 2: Cylinder
  int type;
  Sophus::SE3f T_self_world;
  std::vector<float> parameters;
  GLuint textureId;

  bool isMatte = false;
};

void SetObjectUniforms(
    const GLuint shader,
    const Camera& camera,
    const Light& lighting,
    const SDFObject& object) {
  glUseProgram(shader);

  // Camera parameters
  const Eigen::Matrix3f& K = camera.K;
  const Eigen::Matrix4f T_world_camera = camera.T_world_self.matrix();
  glUniformMatrix3fv(
      glGetUniformLocation(shader, "K"), 1, false, camera.K.data());
  glUniformMatrix4fv(
      glGetUniformLocation(shader, "T_world_camera"),
      1,
      false,
      T_world_camera.data());

  // Lighting parameters
  const Eigen::Matrix4f T_light_world = lighting.T_self_world.matrix();
  glUniformMatrix4fv(
      glGetUniformLocation(shader, "T_light_world"),
      1,
      false,
      T_light_world.data());
  glUniform1i(glGetUniformLocation(shader, "isMatte"), object.isMatte);

  // Object parameters
  const Eigen::Matrix4f T_shape_world = object.T_self_world.matrix();
  glUniform1i(glGetUniformLocation(shader, "shapeType"), object.type);
  glUniformMatrix4fv(
      glGetUniformLocation(shader, "T_shape_world"),
      1,
      false,
      T_shape_world.data());

  std::vector<float> params = object.parameters;
  if (params.size() % 4 != 0) {
    params.resize(params.size() + (4 - params.size() % 4), 0);
  }
  glUniform4fv(
      glGetUniformLocation(shader, "shapeParameters"),
      params.size() / 4,
      params.data());

  glBindTexture(GL_TEXTURE_2D, object.textureId);
}

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

  std::vector<std::filesystem::path> shaderFilepaths = {
      "/Users/static/Documents/code/sdfs/shaders/sdf.vert",
      "/Users/static/Documents/code/sdfs/shaders/single_object.frag"};
  std::map<std::string, std::filesystem::file_time_type> lastModifiedTime;
  auto shaderProgram = CreateShaderProgram(
      readFile("/Users/static/Documents/code/sdfs/shaders/sdf.vert"),
      readFile("/Users/static/Documents/code/sdfs/shaders/single_object.frag"));
  for (const auto& shaderPath : shaderFilepaths) {
    lastModifiedTime[shaderPath] = std::filesystem::last_write_time(shaderPath);
  }

  // Get the heliocentric position of Earth (which gives us the position of the
  // Sun relative to Earth)

  SDFObject sun{
      .type = 1,
      .T_self_world = {Sophus::SO3f(), Eigen::Vector3f{0, 0, 0}},
      .parameters = {695.7},
      .textureId = CreateTexture(
          "/Users/static/Documents/code/sdfs/build/assets/sun.jpg"),
      .isMatte = true};
  SDFObject earth{
      .type = 1,
      .T_self_world = {Sophus::SO3f(), Eigen::Vector3f(0, 0, 0)},
      .parameters = {6.371009},
      .textureId = CreateTexture(
          "/Users/static/Documents/code/sdfs/build/assets/earth.jpg"),
      .isMatte = false};
  SDFObject moon{
      .type = 1,
      .T_self_world = {Sophus::SO3f(), Eigen::Vector3f{0, 0, 0}},
      .parameters = {1.7374},
      .textureId = CreateTexture(
          "/Users/static/Documents/code/sdfs/build/assets/moon.jpg"),
      .isMatte = false};

  auto start = std::chrono::high_resolution_clock::now();
  glEnable(GL_DEPTH_TEST);
  SolarSystemPose solar;
  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    bool updateShaderProgram = false;
    for (const auto& path : shaderFilepaths) {
      const auto writeTime = std::filesystem::last_write_time(path);
      if (writeTime != lastModifiedTime.at(path)) {
        lastModifiedTime[path] = writeTime;
        updateShaderProgram = true;
      }
    }
    if (updateShaderProgram) {
      shaderProgram = CreateShaderProgram(
          readFile("/Users/static/Documents/code/sdfs/shaders/sdf.vert"),
          readFile(
              "/Users/static/Documents/code/sdfs/shaders/single_object.frag"));
    }

    // Calculate current time
    const auto timeSecs =
        (start - std::chrono::high_resolution_clock::now()).count() / 1e9;

    // Convert the date to Julian Date
    // long year = 2023;
    // long month = 9;
    // double day = 29;
    // double hour = 0.0;
    // double minute = 0.0;
    // double second = 0.0;
    // CAADate JD(CAADate::DateToJD(year, month, day, true) + timeSecs * 1,
    // true); auto aaSunPosition =
    // CAASun::EclipticRectangularCoordinatesJ2000(JD, true); Eigen::Vector3f
    // sun_world = megameter_to_au *
    //     Eigen::Vector3f{aaSunPosition.X, aaSunPosition.Y, aaSunPosition.Z};

    // // Get the geocentric position of the Moon
    // const Eigen::Vector3f moon_world =
    //     MoonEclipticRectangularCoordinatesJ2000(JD) * 1e-3;

    // sun.T_self_world.translation() = sun_world;
    // moon.T_self_world.translation() = moon_world;
    // const double apparentGST =
    // CAASidereal::ApparentGreenwichSiderealTime(JD); const double rotationRads
    // = (apparentGST / 24.0) * 2 * M_PI; earth.T_self_world = Sophus::SE3f(
    //     Sophus::SO3f::rotY(rotationRads), Eigen::Vector3f{0, 0, 0});

    // Define the UTC time for which you want the data
    ConstSpiceChar* utc = "2021-09-30T12:00:00";

    // Convert the UTC time to ephemeris time (TDB)
    SpiceDouble et;
    str2et_c(utc, &et);

    et += timeSecs * 60 * 60 * 24;
    Sophus::SE3d T_J2000_earth = solar.T_J2000_body("EARTH", et);
    Sophus::SE3d T_J2000_sun = solar.T_J2000_body("SUN", et);
    Sophus::SE3d T_J2000_moon = solar.T_J2000_body("MOON", et);

    // static int x = 0;
    // // if (x++ == 0)
    // {
    //   std::cout << "Earth" << std::endl;
    //   std::cout << T_J2000_earth.translation() << std::endl;
    //   std::cout << "Moon" << std::endl;
    //   std::cout << T_J2000_moon.translation() << std::endl;
    //   std::cout << "Sun" << std::endl;
    //   std::cout << T_J2000_sun.translation() << std::endl;
    // }

    // T_J2000_sun.translation() - T_J2000_earth.translation();
    // T_J2000_moon.translation() - T_J2000_earth.translation();
    // T_J2000_earth.translation() - T_J2000_earth.translation();

    earth.T_self_world = T_J2000_earth.inverse().cast<float>();
    sun.T_self_world = T_J2000_sun.inverse().cast<float>();
    moon.T_self_world = T_J2000_moon.inverse().cast<float>();
    // earth.T_self_world =
    //     (T_J2000_earth.inverse() * T_J2000_earth).cast<float>();
    // sun.T_self_world = (T_J2000_sun.inverse() * T_J2000_earth).cast<float>();
    // moon.T_self_world = (T_J2000_moon.inverse() *
    // T_J2000_earth).cast<float>();

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

    Camera camera{
        .T_world_self =
            Sophus::SE3f(Sophus::SO3f::rotY(0), Eigen::Vector3f{0, 0, 500})
                .inverse(),
        .K = K};

    Light lighting{.T_self_world = sun.T_self_world};

    glUseProgram(shaderProgram);
    // SetObjectUniforms(shaderProgram, camera, lighting, sun);
    // // Draw full-screen quad
    // glBindVertexArray(VAO);
    // glDrawArrays(GL_TRIANGLES, 0, 6);
    // glBindVertexArray(0);
    SetObjectUniforms(shaderProgram, camera, lighting, moon);
    // Draw full-screen quad
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    SetObjectUniforms(shaderProgram, camera, lighting, earth);
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
