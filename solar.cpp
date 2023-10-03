#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <GL/glew.h>
#include <Sophus/se3.hpp>
#include <fmt/format.h>

#include "cspice/include/SpiceUsr.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include "imgui/imgui.h"

#include <sophus/se3.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GLFW/glfw3.h>

std::optional<double> intersect(
    const Eigen::Vector3d p,
    const Eigen::Vector3d& d,
    const Eigen::Vector3d& sphere,
    const double radius) {
  Eigen::Vector3d oc = p - sphere;
  double a = d.dot(d);
  double b = 2.0f * oc.dot(d);
  double c = oc.dot(oc) - radius * radius;

  double discriminant = b * b - 4 * a * c;

  if (discriminant < 0) {
    return {}; // No intersection.
  } else {
    // Return the nearest intersection point.
    return (-b - std::sqrt(discriminant)) / (2.0f * a);
  }
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
  stbi_set_flip_vertically_on_load(true);
  int width, height, nrChannels;
  unsigned char* data =
      stbi_load(filepath.c_str(), &width, &height, &nrChannels, 0);

  const auto texture = CreateTexture(data, width, height, nrChannels);
  stbi_image_free(data);
  return texture;
}

class FileModifiedTracker {
 public:
  FileModifiedTracker(const std::filesystem::path& filepath)
      : filepath_(filepath),
        lastWrite_(std::filesystem::last_write_time(filepath)) {}

  const std::filesystem::path path() const { return filepath_; }

  bool wasModified() const {
    return lastWrite_ != std::filesystem::last_write_time(filepath_);
  }

  void refresh() { lastWrite_ = std::filesystem::last_write_time(filepath_); }

 private:
  std::filesystem::path filepath_;
  std::filesystem::file_time_type lastWrite_;
};

class ReloadableShader {
 public:
  ReloadableShader(
      const std::filesystem::path& vertexShaderFilepath,
      const std::filesystem::path& fragmentShaderFilepath)
      : vertexShader_(vertexShaderFilepath),
        fragmentShader_(fragmentShaderFilepath) {
    loadShaders(vertexShaderFilepath, fragmentShaderFilepath);
  }

  ~ReloadableShader() { glDeleteProgram(shaderProgram_); }

  void loadShaders(
      const std::filesystem::path& vertexShaderFilepath,
      const std::filesystem::path& fragmentShaderFilepath) {
    vertexShader_ = FileModifiedTracker(vertexShaderFilepath);
    fragmentShader_ = FileModifiedTracker(fragmentShaderFilepath);

    const std::string vertexShaderSource =
        readFile(vertexShaderFilepath.c_str());
    const std::string fragmentShaderSource =
        readFile(fragmentShaderFilepath.c_str());

    const char* pVertexShaderSource = vertexShaderSource.c_str();
    const char* pFragmentShaderSource = fragmentShaderSource.c_str();

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &pVertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &pFragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    shaderProgram_ = glCreateProgram();

    glAttachShader(shaderProgram_, vertexShader);
    glAttachShader(shaderProgram_, fragmentShader);
    glLinkProgram(shaderProgram_);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
  }

  GLuint id() {
    maybeReload();
    return shaderProgram_;
  }

 private:
  void maybeReload() {
    if (vertexShader_.wasModified() || fragmentShader_.wasModified()) {
      glDeleteProgram(shaderProgram_);
      loadShaders(vertexShader_.path(), fragmentShader_.path());
    }
  }

  FileModifiedTracker vertexShader_;
  FileModifiedTracker fragmentShader_;
  GLuint shaderProgram_;
};

class ReloadableTexture {
 public:
  ReloadableTexture(unsigned char r, unsigned char g, unsigned char b) {
    texture_ = CreateTexture(r, g, b);
  }

  ReloadableTexture(const std::filesystem::path& filepath)
      : filepath_(filepath) {
    load(filepath);
  }

  void load(const std::filesystem::path& path) {
    filepath_ = FileModifiedTracker(path);
    release();
    texture_ = CreateTexture(filepath_->path());
  }

  void maybeReload() {
    if (filepath_ && filepath_->wasModified()) {
      release();
      texture_ = CreateTexture(filepath_->path());
    }
  }

  GLuint id() { return texture_; }

  void release() {
    texture_ = 0;
    glDeleteTextures(1, &texture_);
  }

 private:
  mutable GLuint texture_;
  std::optional<FileModifiedTracker> filepath_;
};

class ImguiOpenGLRenderer {
 public:
  ImguiOpenGLRenderer(const std::string& title)
      : title_(title), width_(1.0f), height_(1.0f) {
    // Create the framebuffer
    glGenFramebuffers(1, &framebuffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);

    // Create the texture to hold color info
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_, 0);

    // Create the depth and stencil buffer
    glGenRenderbuffers(1, &depthStencilBuffer_);
    glBindRenderbuffer(GL_RENDERBUFFER, depthStencilBuffer_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 1, 1);
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER,
        GL_DEPTH_STENCIL_ATTACHMENT,
        GL_RENDERBUFFER,
        depthStencilBuffer_);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      throw std::runtime_error("Framebuffer not complete");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  ~ImguiOpenGLRenderer() {
    glDeleteFramebuffers(1, &framebuffer_);
    glDeleteTextures(1, &texture_);
    glDeleteRenderbuffers(1, &depthStencilBuffer_);
  }

  ImVec2 size() { return {width_, height_}; }

  void bind() {
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);

    const ImGuiWindowFlags windowFlags =
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

    ImGui::Begin(title_.c_str(), nullptr, windowFlags);
    const ImVec2 size = ImGui::GetWindowSize();

    if (static_cast<int>(size.x) != width_ ||
        static_cast<int>(size.y) != height_) {
      resizeAttachments(static_cast<int>(size.x), static_cast<int>(size.y));
    }
    glViewport(0, 0, width_, height_);
  }

  void clear() { glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); }

  void unbind() {
    ImGui::Image(
        reinterpret_cast<void*>(static_cast<intptr_t>(texture_)),
        {width_, height_});
    ImGui::End();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

 private:
  void resizeAttachments(int newWidth, int newHeight) {
    // Resize color texture
    glBindTexture(GL_TEXTURE_2D, texture_);
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
    glBindRenderbuffer(GL_RENDERBUFFER, depthStencilBuffer_);
    glRenderbufferStorage(
        GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, newWidth, newHeight);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    width_ = newWidth;
    height_ = newHeight;
  }

  const std::string title_;
  GLuint framebuffer_;
  GLuint texture_;
  GLuint depthStencilBuffer_;
  float width_;
  float height_;
};

class SpiceKernelPackage {
 public:
  SpiceKernelPackage(const std::vector<std::filesystem::path>& kernels)
      : kernels_(kernels) {
    for (const auto& path : kernels_) {
      furnsh_c(path.c_str());
    }
  }

  ~SpiceKernelPackage() {
    for (const auto& path : kernels_) {
      unload_c(path.c_str());
    }
  }

  const std::vector<std::filesystem::path> paths() const { return kernels_; }

 private:
  std::vector<std::filesystem::path> kernels_;
};

class SpiceHelper {
 public:
  static SpiceHelper& getInstance() {
    static SpiceHelper instance;
    return instance;
  }

  static Sophus::SO3d R_J2000_body(
      const std::string& body, double ephemerisTime) {
    SpiceDouble R_J2000_body[3][3];
    pxform_c(body.c_str(), "J2000", ephemerisTime, R_J2000_body);

    SpiceDouble q_J2000_body[4];
    m2q_c(R_J2000_body, q_J2000_body);

    return Sophus::SO3d(Eigen::Quaterniond(
        q_J2000_body[0], q_J2000_body[1], q_J2000_body[2], q_J2000_body[3]));
  }

  static Eigen::Vector3d position_J2000(
      const std::string& body, double ephemerisTime) {
    SpiceDouble bodyState[6], lt;
    spkezr_c(body.c_str(), ephemerisTime, "J2000", "NONE", "0", bodyState, &lt);
    return Eigen::Vector3d(bodyState[0], bodyState[1], bodyState[2]) / 1e3;
  }

  static Sophus::SE3d T_J2000_body(
      const std::string& body, double ephemerisTime) {
    return Sophus::SE3d(
        R_J2000_body(body, ephemerisTime), position_J2000(body, ephemerisTime));
  }

  static Eigen::Vector3d Radii(const std::string& body) {
    SpiceInt dim;
    SpiceDouble radiiiKm[3];

    bodvrd_c(body.c_str(), "RADII", 3, &dim, radiiiKm);
    return Eigen::Vector3d(radiiiKm[0], radiiiKm[1], radiiiKm[2]) / 1e3;
  }

  static SpiceDouble EphemerisTimeNow() {
    // Get the current UTC time
    std::time_t t = std::time(nullptr);
    char utc_time[40];
    std::strftime(
        utc_time, sizeof(utc_time), "%Y-%m-%dT%H:%M:%S", std::gmtime(&t));

    // Convert UTC time to Ephemeris Time (ET)
    SpiceDouble et;
    utc2et_c(utc_time, &et);

    return et;
  }

  static std::string EphemerisTimeToDate(SpiceDouble et) {
    char utc_time[40];
    et2utc_c(et, "C", 0, 40, utc_time);
    return std::string(utc_time);
  }

  static SpiceDouble EphemerisTimeFromDate(
      int year,
      int month,
      int day,
      int hours = 0,
      int minutes = 0,
      int seconds = 0) {
    // Ensure the instance is instantiated for kernels
    (void)getInstance();

    // Format the input date into an ISO 8601 string
    char utc_time[40];
    snprintf(
        utc_time,
        sizeof(utc_time),
        "%04d-%02d-%02dT%02d:%02d:%02d",
        year,
        month,
        day,
        hours,
        minutes,
        seconds);

    // Convert UTC time string to Ephemeris Time (ET)
    SpiceDouble et;
    utc2et_c(utc_time, &et);

    return et;
  }

  static const SpiceKernelPackage Kernels;
};

const SpiceKernelPackage SpiceHelper::Kernels = {{
    std::filesystem::path(__FILE__).parent_path() /
        "cspice/kernels/naif0012.tls",
    std::filesystem::path(__FILE__).parent_path() / "cspice/kernels/de440.bsp",
    std::filesystem::path(__FILE__).parent_path() /
        "cspice/kernels/pck00011.tpc",
}};

class PerPixelVAO {
 public:
  PerPixelVAO() {
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

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(
        0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
  }

  ~PerPixelVAO() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
  }

  GLuint id() const { return VAO; }

 private:
  GLuint VAO;
  GLuint VBO;
};

Sophus::SE3d LookAt(
    const Eigen::Vector3d& position_world,
    const Eigen::Vector3d& target_world,
    const Eigen::Vector3d& upHint) {
  const Eigen::Vector3d forward = (target_world - position_world).normalized();
  const Eigen::Vector3d right = (upHint.cross(forward)).normalized();
  const Eigen::Vector3d up = (forward.cross(right)).normalized();
  Eigen::Matrix3d R;
  R.col(0) = -right;
  R.col(1) = -up;
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

class OpenGLWindow {
 public:
  OpenGLWindow(
      const std::string& windowName,
      int width,
      int height,
      bool stayOnTop = false) {
    if (!glfwInit()) {
      throw std::runtime_error("Failed to initilize GLFW.");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(width, height, windowName.c_str(), NULL, NULL);
    if (!window_) {
      glfwTerminate();
      throw std::runtime_error("Failed to create GLFW window.");
    }
    if (stayOnTop) {
      glfwSetWindowAttrib(window_, GLFW_FLOATING, GLFW_TRUE);
    }

    glfwMakeContextCurrent(window_);
    glfwSetKeyCallback(window_, keyCallback);

    if (glewInit() != GLEW_OK) {
      throw std::runtime_error("Failed to initialize GLEW.");
    }

    // Initialize ImGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |=
        ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |=
        ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // IF using Docking Branch
    // ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 410");
  }

  ~OpenGLWindow() {
    // Deletes all ImGUI instances
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
  }

  bool shouldClose() const { return glfwWindowShouldClose(window_); }
  void setErrorCallback(GLFWerrorfun fn) const { glfwSetErrorCallback(fn); }
  void setKeyCallback(GLFWkeyfun fn) const { glfwSetKeyCallback(window_, fn); }

  void beginNewFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuiID dockspace_id =
        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
  }

  void finishFrame() {
    // Renders the ImGUI elements
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window_);
    glfwPollEvents();
  }

 private:
  GLFWwindow* window_;
};

struct Camera {
  Sophus::SE3d T_world_self;
  Eigen::Matrix3f K;
  Eigen::Vector2f resolution;
  bool orthographic;
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
  GLuint texture;

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
  glUniform2fv(
      glGetUniformLocation(shader, "resolution"), 1, camera.resolution.data());
  glUniformMatrix3fv(
      glGetUniformLocation(shader, "K"), 1, false, camera.K.data());
  glUniform1i(
      glGetUniformLocation(shader, "orthographic"), int(camera.orthographic));
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

  glBindTexture(GL_TEXTURE_2D, object.texture);
}

namespace {

struct PoseGroup {
  double et;
  Sophus::SE3d T_world_sun;
  Sophus::SE3d T_world_earth;
  Sophus::SE3d T_world_moon;
};

void TestDistance(const std::vector<PoseGroup>& yearTrajectory) {
  {
    double lo = std::numeric_limits<double>::infinity();
    double hi = 0.0;
    double total = 0.0;
    for (const auto& [et, T_world_sun, T_world_earth, _] : yearTrajectory) {
      const double distance =
          (T_world_sun.inverse() * T_world_earth).translation().norm() / 1e3;
      total += distance;
      lo = std::min(lo, distance);
      hi = std::max(hi, distance);
    }
    fmt::println(
        "Distance to sun: min {:.2f}, max {:.2f}, avg {:.2f}",
        lo,
        hi,
        total / yearTrajectory.size());
  }

  {
    double lo = std::numeric_limits<double>::infinity();
    double hi = 0.0;
    double total = 0.0;
    for (const auto& [et, _, T_world_earth, T_world_moon] : yearTrajectory) {
      const double distance =
          (T_world_moon.inverse() * T_world_earth).translation().norm() / 1e3;
      total += distance;
      lo = std::min(lo, distance);
      hi = std::max(hi, distance);
    }
    fmt::println(
        "Distance to moon: min {:.2f}, max {:.2f}, avg {:.2f}",
        lo,
        hi,
        total / yearTrajectory.size());
  }
}

Eigen::Vector4d fitPlane(const std::vector<Eigen::Vector3d>& points) {
  // Check if we have enough points
  if (points.size() < 3) {
    throw std::runtime_error("Need at least three points to fit a plane");
  }

  // Calculate centroid
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto& pt : points) {
    centroid += pt;
  }
  centroid /= points.size();

  // Assemble the data matrix
  Eigen::MatrixXd A(3, points.size());
  for (size_t i = 0; i < points.size(); i++) {
    A.col(i) = points[i] - centroid;
  }

  // Compute the covariance matrix
  Eigen::MatrixXd C = A * A.transpose();

  // Perform Eigen decomposition
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(C);
  Eigen::Vector3d planeNormal = solver.eigenvectors().col(0);

  // Plane equation: ax + by + cz + d = 0
  // We know the normal is (a, b, c) and a point on the plane (the centroid)
  // So, d = -normal.dot(centroid)
  double d = -planeNormal.dot(centroid);

  return Eigen::Vector4d(planeNormal(0), planeNormal(1), planeNormal(2), d);
}

void TestOrbitTilt(const std::vector<PoseGroup>& yearTrajectory) {
  std::vector<Eigen::Vector3d> earthOrbitAroundSun;
  for (const auto& [_, T_world_sun, T_world_earth, __] : yearTrajectory) {
    earthOrbitAroundSun.push_back(
        T_world_sun.inverse() * T_world_earth.translation());
  }

  const auto& pose = yearTrajectory.front();

  const Eigen::Vector3d earthOrbitNormal =
      fitPlane(earthOrbitAroundSun).head<3>().normalized();
  fmt::println(
      "Earth Orbit Tilt: {:.2f}",
      180 / M_PI *
          std::acos(earthOrbitNormal.dot(
              pose.T_world_sun.so3().inverse() * pose.T_world_earth.so3() *
              Eigen::Vector3d::UnitZ())));
}

void RunUnitTests() {
  constexpr double secondsInDay = 60 * 60 * 24;
  constexpr double secondsInYear = secondsInDay * 365.25;
  constexpr auto startDate = 750578468.1823622;

  // Collect poses every hour for the past year
  std::vector<PoseGroup> yearTrajectory;
  for (double et = startDate; et < startDate + secondsInYear;
       et += secondsInDay) {
    yearTrajectory.push_back({
        et,
        .T_world_sun = SpiceHelper::T_J2000_body("SUN", et),
        .T_world_earth = SpiceHelper::T_J2000_body("EARTH", et),
        .T_world_moon = SpiceHelper::T_J2000_body("MOON", et),
    });
  }

  TestDistance(yearTrajectory);
  TestOrbitTilt(yearTrajectory);
}
} // namespace

class SolarSystemState {
 public:
  enum BodyId { J2000 = -1, EARTH, MOON, MARS, SUN };

  struct PlanetState {
    BodyId id;
    double radius;
  };

  SolarSystemState() : ephemerisTime_(0.0f), origin_(SUN) {
    const std::vector<BodyId> bodies = {EARTH, MOON, MARS, SUN};
    for (const auto body : bodies) {
      bodies_[body] = PlanetState{
          .id = body,
          .radius = SpiceHelper::Radii(BodyNames[body][2]).mean(),
      };
    }
  }

  void setTime(double newEphemerisTime) {
    ephemerisTime_ = newEphemerisTime;
    setOrigin(origin_); // Refresh origin and its adjacent members
  }

  double getTime() const { return ephemerisTime_; }

  void setOrigin(BodyId body) {
    origin_ = body;
    T_origin_J2000 = T_J2000_body(body).inverse();
  }

  BodyId origin() const { return origin_; }

  double radius(BodyId id) const { return bodies_.at(id).radius; }

  Sophus::SE3d T_origin_body(BodyId id) const {
    return T_origin_J2000 * T_J2000_body(id);
  }

  Eigen::Vector3d bodyLla_origin(
      BodyId id, double longitude, double latitude, double altitude) const {
    Eigen::Vector3d position_body;
    latrec_c(bodies_.at(id).radius, longitude, latitude, position_body.data());
    return T_origin_body(id) * position_body;
  }

  const std::map<BodyId, PlanetState>& bodies() { return bodies_; }

 private:
  // Stored as: {position_name, rotation_name, radii_name}
  static constexpr const char* BodyNames[4][3] = {
      {"EARTH", "IAU_EARTH", "EARTH"},
      {"MOON", "IAU_MOON", "MOON"},
      {"MARS BARYCENTER", "IAU_MARS", "MARS"},
      {"SUN", "IAU_SUN", "SUN"},
  };

  Sophus::SE3d T_J2000_body(BodyId id) const {
    const auto index = static_cast<int>(id);
    return Sophus::SE3d(
        SpiceHelper::R_J2000_body(BodyNames[index][1], ephemerisTime_),
        SpiceHelper::position_J2000(BodyNames[index][0], ephemerisTime_));
  }

  double ephemerisTime_;
  BodyId origin_;
  Sophus::SE3d T_origin_J2000;
  std::map<BodyId, PlanetState> bodies_;
};

int main() {
  // RunUnitTests();
  const std::filesystem::path parentDirectory =
      std::filesystem::path(__FILE__).parent_path();

  OpenGLWindow window("Solar System", 1980, 1080, true);
  PerPixelVAO vao;
  ReloadableShader shader(
      parentDirectory / "shaders/sdf.vert",
      parentDirectory / "shaders/single_object.frag");

  SolarSystemState systemState;

  std::map<SolarSystemState::BodyId, ReloadableTexture> systemTextures{
      {SolarSystemState::SUN,
       ReloadableTexture(parentDirectory / "assets/8k_sun.jpg")},
      {SolarSystemState::EARTH,
       ReloadableTexture(parentDirectory / "assets/earth.jpg")},
      {SolarSystemState::MOON,
       ReloadableTexture(parentDirectory / "assets/moon.jpg")},
      {SolarSystemState::MARS, ReloadableTexture(173, 98, 66)}};
  ImguiOpenGLRenderer gameTab("Game");

  float daysPerSecond = 0;
  float cameraFieldOfView = 2.0f / 180.0 * M_PI; // 120.0f / 180.0 * M_PI;
  Eigen::Vector3f lla{47.608013 / 180 * M_PI, -122.335167 / 180 * M_PI, 3};

  bool hideEarth = false;
  bool isOrthographic = false;
  bool reverseTime = false;
  int ymdhms[6] = {2023, 10, 14, 14, 15, 0};

  systemState.setTime(SpiceHelper::EphemerisTimeFromDate(
      ymdhms[0], ymdhms[1], ymdhms[2], ymdhms[3], ymdhms[4], ymdhms[5]));

  auto previousTime = std::chrono::high_resolution_clock::now();
  glEnable(GL_DEPTH_TEST);
  while (!window.shouldClose()) {
    window.beginNewFrame();

    const char* lookatOptions[] = {
        "Earth", "Horizon", "Moon", "Sun", "EarthTopDown", "SunTopDown"};
    static int currentLook = 2;

    const char* originOptions[] = {"J2000", "Earth", "Moon", "Sun"};
    static int currentOrigin = 0;

    // ImGUI window creation
    ImGui::Begin("Controls");
    const auto dateStr =
        SpiceHelper::EphemerisTimeToDate(systemState.getTime());
    ImGui::Text("Current Date: %s", dateStr.c_str());
    ImGui::Text("Time:");
    ImGui::SliderFloat(
        "Days per Second",
        &daysPerSecond,
        0,
        31,
        "%.3f",
        ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Reverse time", &reverseTime);
    ImGui::Text("Modify Date (UTC):");
    ImGui::InputInt3("Year/Month/Day", ymdhms);
    ImGui::InputInt3("Hour/Minute/Second", ymdhms + 3);
    if (ImGui::Button("Apply")) {
      systemState.setTime(SpiceHelper::EphemerisTimeFromDate(
          ymdhms[0], ymdhms[1], ymdhms[2], ymdhms[3], ymdhms[4], ymdhms[5]));
    }

    // Text that appears in the window
    ImGui::Text("Camera Position:");
    // Slider that appears in the window
    ImGui::Combo(
        "Origins", &currentOrigin, originOptions, IM_ARRAYSIZE(originOptions));
    ImGui::SliderAngle("Latitude", &lla.x(), -90, 90);
    ImGui::SliderAngle("Longitude", &lla.y(), -180, 180);
    if (std::string(lookatOptions[currentLook]) ==
        std::string("EarthTopDown")) {
      ImGui::SliderFloat("Altitude (km)", &lla.z(), 1.0f, 1e6f);
    } else if (std::string(lookatOptions[currentLook]) == "SunTopDown") {
      ImGui::SliderFloat("Altitude (km)", &lla.z(), 1.0f, 200e6f);
    } else {
      ImGui::SliderFloat("Altitude (km)", &lla.z(), 1.0f, 1e4);
    }
    ImGui::Text("Camera Settings:");
    ImGui::Checkbox("Orthographic", &isOrthographic);
    ImGui::SliderAngle(
        "Vertical FoV",
        &cameraFieldOfView,
        0,
        120.0f,
        "%.0f deg",
        ImGuiSliderFlags_Logarithmic);
    ImGui::Combo(
        "Look At", &currentLook, lookatOptions, IM_ARRAYSIZE(lookatOptions));
    ImGui::Checkbox("Hide Earth", &hideEarth);
    ImGui::End();

    // Calculate current time
    const auto currentTime = std::chrono::high_resolution_clock::now();
    auto dtSecs = (currentTime - previousTime).count() / 1e9;
    if (reverseTime) {
      dtSecs *= -1;
    }
    previousTime = currentTime;

    systemState.setTime(
        systemState.getTime() + dtSecs * 60 * 60 * 24 * daysPerSecond);

    Sophus::SE3d T_earth_camera;

    const std::string chosenLook = lookatOptions[currentLook];
    const auto R = systemState.bodies().at(SolarSystemState::EARTH).radius;
    const auto lat = lla.x();
    const auto lon = lla.y();
    const auto alt = lla.z() / 1e3;

    Eigen::Vector3d camera_earth;
    latrec_c(R, lon, lat, camera_earth.data());
    camera_earth += camera_earth.normalized() * alt;

    const std::string origin = originOptions[currentOrigin];
    if (origin == "Earth") {
      systemState.setOrigin(SolarSystemState::EARTH);
    } else if (origin == "Moon") {
      systemState.setOrigin(SolarSystemState::MOON);
    } else if (origin == "Sun") {
      systemState.setOrigin(SolarSystemState::SUN);
    }

    if (chosenLook == "Earth") {
      T_earth_camera = LookAt(
          camera_earth, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
    } else if (chosenLook == "Horizon") {
      T_earth_camera = LookAt(
          camera_earth,
          camera_earth + Eigen::Vector3d::UnitY(),
          -camera_earth.normalized());
    } else if (chosenLook == "Moon") {
      const Sophus::SE3d T_earth_moon =
          systemState.T_origin_body(SolarSystemState::EARTH).inverse() *
          systemState.T_origin_body(SolarSystemState::MOON);
      T_earth_camera = LookAt(
          camera_earth,
          T_earth_moon.translation(),
          T_earth_moon.so3() * Eigen::Vector3d::UnitZ());
    } else if (chosenLook == "Sun") {
      const Sophus::SE3d T_earth_sun =
          systemState.T_origin_body(SolarSystemState::EARTH).inverse() *
          systemState.T_origin_body(SolarSystemState::SUN);
      T_earth_camera = LookAt(
          camera_earth,
          T_earth_sun.translation(),
          T_earth_sun.so3() * Eigen::Vector3d::UnitZ());
    } else if (chosenLook == "EarthTopDown") {
      T_earth_camera = LookAt(
          Eigen::Vector3d::UnitZ() *
              (systemState.radius(SolarSystemState::EARTH) + alt),
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::UnitY());
    } else if (chosenLook == "SunTopDown") {
      const Sophus::SE3d T_earth_sun =
          systemState.T_origin_body(SolarSystemState::EARTH).inverse() *
          systemState.T_origin_body(SolarSystemState::SUN);

      T_earth_camera = LookAt(
          T_earth_sun * Eigen::Vector3d::UnitZ() * (alt),
          T_earth_sun * Eigen::Vector3d::Zero(),
          Eigen::Vector3d::UnitY());
    }

    gameTab.bind();
    gameTab.clear();
    const ImVec2 size = gameTab.size();
    const auto [width, height] = size;

    const auto fy = (height / 2.0) / tan(cameraFieldOfView / 2.0);
    Eigen::Matrix3f K;
    K << fy, 0, width / 2.0, 0, fy, height / 2.0, 0, 0, 1;

    Camera camera{
        .T_world_self =
            systemState.T_origin_body(SolarSystemState::EARTH) * T_earth_camera,
        .K = K,
        .resolution = {width, height},
        .orthographic = isOrthographic};

    Light lighting{
        .T_self_world =
            systemState.T_origin_body(SolarSystemState::SUN).inverse()};

    const auto createObject = [&](SolarSystemState::BodyId id,
                                  bool isMatte = false) {
      return SDFObject{
          .isMatte = isMatte,
          .parameters = {float(systemState.radius(id))},
          .T_self_world = systemState.T_origin_body(id).inverse(),
          .texture = systemTextures.at(id).id(),
          .type = 1};
    };

    const auto shaderProgram = shader.id();
    glUseProgram(shaderProgram);
    glBindVertexArray(vao.id());
    SetObjectUniforms(
        shaderProgram,
        camera,
        lighting,
        createObject(SolarSystemState::SUN, true));
    glDrawArrays(GL_TRIANGLES, 0, 6);
    SetObjectUniforms(
        shaderProgram, camera, lighting, createObject(SolarSystemState::MOON));
    glDrawArrays(GL_TRIANGLES, 0, 6);
    if (!hideEarth) {
      SetObjectUniforms(
          shaderProgram,
          camera,
          lighting,
          createObject(SolarSystemState::EARTH));
      glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    // if (object) {
    //   fmt::println("Rendering object..");
    //   SetObjectUniforms(shaderProgram, camera, lighting, *object);
    //   glDrawArrays(GL_TRIANGLES, 0, 6);
    // }
    gameTab.unbind();

    window.finishFrame();
  }

  // Clean up textures to be nice
  for (auto& [_, texture] : systemTextures) {
    texture.release();
  }

  return 0;
}
