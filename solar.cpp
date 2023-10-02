#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <GL/glew.h>
#include <Sophus/se3.hpp>
#include <fmt/format.h>
#include "SpiceUsr.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include "imgui/imgui.h"

#include <sophus/se3.hpp>
#include "includes/aaplus/AA+.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GLFW/glfw3.h>

class ImguiOpenGLRenderer {
 private:
  GLuint framebuffer;
  GLuint texture;
  GLuint depthStencilBuffer;
  int width, height;

 public:
  ImguiOpenGLRenderer(int width, int height) : width(width), height(height) {
    // Create the framebuffer
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    // Create the texture to hold color info
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        width,
        height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

    // Create the depth and stencil buffer
    glGenRenderbuffers(1, &depthStencilBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthStencilBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER,
        GL_DEPTH_STENCIL_ATTACHMENT,
        GL_RENDERBUFFER,
        depthStencilBuffer);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      // Handle error
      throw std::runtime_error("Framebuffer not complete");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  ~ImguiOpenGLRenderer() {
    glDeleteFramebuffers(1, &framebuffer);
    glDeleteTextures(1, &texture);
    glDeleteRenderbuffers(1, &depthStencilBuffer);
  }

  void resizeAttachments(int newWidth, int newHeight) {
    // Resize color texture
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        newWidth,
        newHeight,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Resize depth and stencil renderbuffer
    glBindRenderbuffer(GL_RENDERBUFFER, depthStencilBuffer);
    glRenderbufferStorage(
        GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, newWidth, newHeight);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    width = newWidth;
    height = newHeight;
  }

  void render(
      const std::string& title, const std::function<void(ImVec2)>& renderFn) {
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    const ImGuiWindowFlags windowFlags =
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

    ImGui::Begin(title.c_str(), nullptr, windowFlags);
    const ImVec2 size = ImGui::GetWindowSize();

    if (static_cast<int>(size.x) != width ||
        static_cast<int>(size.y) != height) {
      resizeAttachments(static_cast<int>(size.x), static_cast<int>(size.y));
    }

    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    renderFn(size);

    ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture)), size);
    ImGui::End();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }
};

class SPICEPose {
 public:
  SPICEPose() { init(); }
  ~SPICEPose() { cleanup(); }

  Sophus::SE3d T_J2000_body(const std::string& body, double ephemerisTime) {
    SpiceDouble bodyState[6], lt;
    spkezr_c(
        body.c_str(), ephemerisTime, "J2000", "NONE", "EARTH", bodyState, &lt);

    const std::string& fixedFrame = "IAU_" + body;
    SpiceDouble R_J2000_body[3][3];
    pxform_c(fixedFrame.c_str(), "J2000", ephemerisTime, R_J2000_body);
    SpiceDouble q_J2000_body[4];
    m2q_c(R_J2000_body, q_J2000_body);

    Eigen::Quaterniond J2000_body_xzy(
        q_J2000_body[0], q_J2000_body[1], q_J2000_body[2], q_J2000_body[3]);
    return Sophus::SE3d(
        J2000_body_xzy.matrix(),
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

Sophus::SE3d LookAt(
    const Eigen::Vector3d& position_world,
    const Eigen::Vector3d& target_world,
    const Eigen::Vector3d& upHint) {
  const Eigen::Vector3d forward = (target_world - position_world).normalized();
  const Eigen::Vector3d right = (upHint.cross(forward)).normalized();
  const Eigen::Vector3d up = (forward.cross(right)).normalized();
  Eigen::Matrix3d R;
  R.col(0) = right;
  R.col(1) = up;
  R.col(2) = forward;
  return Sophus::SE3d(R, position_world);
};

static void error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error: %s\n", description);
}

void keyCallback(
    GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
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
  if (!std::filesystem::exists(filepath)) {
    fmt::println("Couldn't find texture path: {}", filepath);
    exit(-1);
  }
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
  Sophus::SE3d T_world_self;
  Eigen::Matrix3f K;
};

struct Light {
  Sophus::SE3d T_self_world;
  //   Eigen::Vector3f color;
};

struct SDFObject {
  // 1: Sphere, 2: Cylinder
  int type;
  Sophus::SE3d T_self_world;
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
  const Eigen::Matrix4f T_shape_camera =
      (object.T_self_world * camera.T_world_self).matrix().cast<float>();
  glUniformMatrix3fv(
      glGetUniformLocation(shader, "K"), 1, false, camera.K.data());
  glUniformMatrix4fv(
      glGetUniformLocation(shader, "T_shape_camera"),
      1,
      false,
      T_shape_camera.data());

  // Lighting parameters
  const Eigen::Vector3f light_shape =
      (object.T_self_world * lighting.T_self_world.inverse())
          .translation()
          .cast<float>();
  glUniform3fv(
      glGetUniformLocation(shader, "light_shape"), 1, light_shape.data());
  glUniform1i(glGetUniformLocation(shader, "isMatte"), object.isMatte);

  // Object parameters
  const Eigen::Matrix4f T_shape_world =
      object.T_self_world.matrix().cast<float>();
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

  GLFWwindow* window = glfwCreateWindow(
      640, 640 * 20 / 9, "Solar System - Raymarching SDF", NULL, NULL);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwSetWindowAttrib(window, GLFW_FLOATING, GLFW_TRUE);

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

  // Initialize ImGUI
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |=
      ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
  io.ConfigFlags |=
      ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // IF using Docking Branch
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

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
      .T_self_world = {},
      .parameters = {696.34},
      .textureId =
          CreateTexture("/Users/static/Documents/code/sdfs/assets/8k_sun.jpg"),
      .isMatte = true};
  SDFObject earth{
      .type = 1,
      .T_self_world = {},
      .parameters = {6.371009},
      .textureId =
          CreateTexture("/Users/static/Documents/code/sdfs/assets/earth.jpg"),
      .isMatte = false};
  SDFObject moon{
      .type = 1,
      .T_self_world = {},
      .parameters = {1.7374},
      .textureId =
          CreateTexture("/Users/static/Documents/code/sdfs/assets/moon2.jpg"),
      .isMatte = false};

  float daysPerSecond = 2.5; // .1;
  float cameraFieldOfView = 1.0f / 180.0 * M_PI;
  Eigen::Vector3f lla{47.608013 / 180 * M_PI, -122.335167 / 180 * M_PI, 3};

  ImguiOpenGLRenderer imguiRender(960, 540);

  bool hideEarth = false;
  // earth, up, moon, sun
  bool lookat[4] = {false, false, false, false};
  auto start = std::chrono::high_resolution_clock::now();
  glEnable(GL_DEPTH_TEST);
  SPICEPose solar;

  // Convert the UTC time to ephemeris time (TDB)
  // ConstSpiceChar* utc2 = "2021-03-30T12:00:00";
  // SpiceDouble et2;
  // str2et_c(utc2, &et2);
  // solar.T_J2000_body("MARS", 0);
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

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    const char* lookatOptions[] = {
        "Earth", "Horizon", "Moon", "Sun", "EarthTopDown", "SunTopDown"};
    static int currentLook = 2;

    const char* originOptions[] = {"J2000", "Earth", "Moon", "Sun"};
    static int currentOrigin = 0;

    // ImGUI window creation
    ImGui::Begin("Controls");
    ImGui::Text("Time:");
    ImGui::SliderFloat("Days per Second", &daysPerSecond, 0, 31);
    // Text that appears in the window
    ImGui::Text("Camera Position:");
    // Slider that appears in the window
    ImGui::Combo(
        "Origins", &currentOrigin, originOptions, IM_ARRAYSIZE(originOptions));
    ImGui::SliderAngle("Latitude", &lla.x(), -90.0f, 90.0f);
    ImGui::SliderAngle("Longitude", &lla.y(), -180.0f, 180.0f);
    if (std::string(lookatOptions[currentLook]) ==
        std::string("EarthTopDown")) {
      ImGui::SliderFloat("Altitude (km)", &lla.z(), 1.0f, 1e6f);
    } else if (std::string(lookatOptions[currentLook]) == "SunTopDown") {
      ImGui::SliderFloat("Altitude (km)", &lla.z(), 1.0f, 200e6f);
    } else {
      ImGui::SliderFloat("Altitude (km)", &lla.z(), 1.0f, 5e4f);
    }
    ImGui::Text("Camera Settings:");
    ImGui::SliderAngle("Vertical FoV", &cameraFieldOfView, 0, 120.0f);
    ImGui::Combo(
        "Look At", &currentLook, lookatOptions, IM_ARRAYSIZE(lookatOptions));
    ImGui::Checkbox("Hide Earth", &hideEarth);

    // Ends the window
    ImGui::End();
    // ImGui::Begin("DockSpace Demo"); // Create a new window
    // ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    // ImGui::DockSpace(dockspace_id);
    // ImGui::End();

    // Calculate current time
    const auto timeSecs =
        (start - std::chrono::high_resolution_clock::now()).count() / 1e9;

    // Define the UTC time for which you want the data
    ConstSpiceChar* utc = "2021-03-30T12:00:00";

    // Convert the UTC time to ephemeris time (TDB)
    SpiceDouble et;
    str2et_c(utc, &et);

    et += timeSecs * 60 * 60 * 24 * daysPerSecond;
    Sophus::SE3d T_J2000_earth = solar.T_J2000_body("EARTH", et);
    Sophus::SE3d T_J2000_sun = solar.T_J2000_body("SUN", et);
    Sophus::SE3d T_J2000_moon = solar.T_J2000_body("MOON", et);

    earth.T_self_world = T_J2000_earth.inverse();
    sun.T_self_world = T_J2000_sun.inverse();
    moon.T_self_world = T_J2000_moon.inverse();

    const std::string origin = originOptions[currentOrigin];
    Sophus::SE3d T_world_newWorld;
    if (origin == "Earth") {
      T_world_newWorld = earth.T_self_world.inverse();
    } else if (origin == "Moon") {
      T_world_newWorld = moon.T_self_world.inverse();
    } else if (origin == "Sun") {
      T_world_newWorld = sun.T_self_world.inverse();
    }
    earth.T_self_world *= T_world_newWorld;
    sun.T_self_world *= T_world_newWorld;
    moon.T_self_world *= T_world_newWorld;

    Sophus::SE3d T_earth_camera;

    const std::string chosenLook = lookatOptions[currentLook];
    const auto R = earth.parameters.at(0);
    const auto lat = lla.x();
    const auto lon = -lla.y();
    const auto alt = lla.z() / 1e3;
    Eigen::Vector3d camera_earth = {
        (R + alt) * std::cos(lat) * std::cos(lon),
        (R + alt) * std::cos(lat) * std::sin(lon),
        (R + alt) * std::sin(lat)};
    if (chosenLook == "Earth") {
      T_earth_camera = LookAt(
          camera_earth, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
    } else if (chosenLook == "Horizon") {
      T_earth_camera = LookAt(
          camera_earth,
          camera_earth + Eigen::Vector3d::UnitY(),
          camera_earth.normalized());
    } else if (chosenLook == "Moon") {
      const Sophus::SE3d T_earth_moon =
          earth.T_self_world * moon.T_self_world.inverse();
      T_earth_camera = LookAt(
          camera_earth,
          T_earth_moon.translation(),
          T_earth_moon.so3() * Eigen::Vector3d::UnitZ());
    } else if (chosenLook == "Sun") {
      const Sophus::SE3d T_earth_sun =
          earth.T_self_world * sun.T_self_world.inverse();
      T_earth_camera = LookAt(
          camera_earth,
          T_earth_sun.translation(),
          T_earth_sun.so3() * Eigen::Vector3d::UnitZ());
    } else if (chosenLook == "EarthTopDown") {
      T_earth_camera = LookAt(
          Eigen::Vector3d::UnitZ() * (earth.parameters.at(0) + alt),
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::UnitY());
    } else if (chosenLook == "SunTopDown") {
      const Sophus::SE3d T_earth_sun =
          earth.T_self_world * sun.T_self_world.inverse();

      T_earth_camera = LookAt(
          T_earth_sun * Eigen::Vector3d::UnitZ() * (alt),
          T_earth_sun * Eigen::Vector3d::Zero(),
          Eigen::Vector3d::UnitY());
    }

    imguiRender.render("Game", [&](const ImVec2& size) {
      const auto [width, height] = size;

      const auto fy = (height / 2.0) / tan(cameraFieldOfView / 2.0);
      Eigen::Matrix3f K;
      K << fy, 0, width / 2.0, 0, fy, height / 2.0, 0, 0, 1;

      Camera camera{
          .T_world_self = earth.T_self_world.inverse() * T_earth_camera,
          .K = K};

      Light lighting{.T_self_world = sun.T_self_world};

      glUseProgram(shaderProgram);
      SetObjectUniforms(shaderProgram, camera, lighting, sun);
      glBindVertexArray(VAO);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glBindVertexArray(0);

      SetObjectUniforms(shaderProgram, camera, lighting, moon);
      glBindVertexArray(VAO);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glBindVertexArray(0);

      if (!hideEarth) {
        SetObjectUniforms(shaderProgram, camera, lighting, earth);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
      }
    });

    // Renders the ImGUI elements
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // Deletes all ImGUI instances
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteProgram(shaderProgram);

  glfwTerminate();
  return 0;
}
