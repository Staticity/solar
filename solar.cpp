#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
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
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
    ImVec2 size = ImGui::GetContentRegionAvail();
    width_ = size.x;
    height_ = size.y;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Image(
        reinterpret_cast<void*>(static_cast<intptr_t>(texture_)),
        ImVec2(width_, height_));
    ImGui::Image(
        reinterpret_cast<void*>(static_cast<intptr_t>(texture_)),
        {width_, height_});
    ImGui::PopStyleVar();
    ImGui::End();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  GLuint texture() { return texture_; }

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
    std::filesystem::path(__FILE__).parent_path() /
        "cspice/kernels/gm_de440.tpc",
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
  if (right.norm() < 1e-6 || up.norm() < 1e-6) {
    return Sophus::SE3d(Sophus::SO3d(), position_world);
  }
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

  std::pair<int, int> getSize() const {
    std::pair<int, int> wh;
    glfwGetFramebufferSize(window_, &wh.first, &wh.second);
    return wh;
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

enum BodyId {
  J2000 = -1,
  EARTH,
  MOON,
  SUN,
  MERCURY,
  VENUS,
  MARS,
  JUPITER,
  URANUS,
  SATURN,
  NEPTUNE
};

std::string toString(BodyId id) {
  if (id == J2000)
    return "J2000";
  if (id == EARTH)
    return "EARTH";
  if (id == MOON)
    return "MOON";
  if (id == SUN)
    return "SUN";
  if (id == MERCURY)
    return "MERCURY";
  if (id == VENUS)
    return "VENUS";
  if (id == MARS)
    return "MARS";
  if (id == JUPITER)
    return "JUPITER";
  if (id == SATURN)
    return "SATURN";
  if (id == URANUS)
    return "URANUS";
  if (id == NEPTUNE)
    return "NEPTUNE";
  return "N/A";
}

class SolarSystemState {
 public:
  struct PlanetState {
    BodyId id;
    double radius;
  };

  SolarSystemState() : ephemerisTime_(0.0f), origin_(SUN) {
    const std::vector<BodyId> bodies = {
        EARTH,
        MOON,
        SUN,
        MARS,
        MERCURY,
        VENUS,
        JUPITER,
        URANUS,
        SATURN,
        NEPTUNE};
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
    if (body == J2000) {
      origin_ = J2000;
      T_origin_J2000 = Sophus::SE3d();
      return;
    }
    origin_ = body;
    T_origin_J2000 = T_J2000_body(body).inverse();
  }

  BodyId origin() const { return origin_; }

  double radius(BodyId id) const { return bodies_.at(id).radius; }

  double gMass(BodyId id) const {
    SpiceDouble gm;
    SpiceInt n;

    const auto index = static_cast<int>(id);
    bodvrd_c(BodyNames[index][0], "GM", 1, &n, &gm);
    return gm;
  }

  Sophus::SE3d T_origin_body(BodyId id) const {
    return T_origin_J2000 * T_J2000_body(id);
  }

  Eigen::Vector3d position_body(
      BodyId id, double latitude, double longitude, double altitude) const {
    Eigen::Vector3d position_body;
    latrec_c(bodies_.at(id).radius, longitude, latitude, position_body.data());
    return position_body + position_body.normalized() * altitude;
  }

  std::pair<Eigen::Vector3d, Eigen::Vector3d> linearVelocityAcceleration(
      BodyId id,
      double latitude,
      double longitude,
      double altitude,
      double dt = 1,
      double scale = 1) const {
    const Eigen::Vector3d lla_body =
        position_body(id, latitude, longitude, altitude);
    const Eigen::Vector3d p0 =
        T_J2000_body(id, ephemerisTime_).so3() * lla_body;
    const Eigen::Vector3d p1 =
        T_J2000_body(id, ephemerisTime_ + dt).so3() * lla_body;
    const Eigen::Vector3d p2 =
        T_J2000_body(id, ephemerisTime_ + dt + dt).so3() * lla_body;

    const Eigen::Vector3d v0 = ((p1 - p0) / dt) * scale;
    const Eigen::Vector3d v1 = ((p2 - p1) / dt) * scale;

    const Eigen::Vector3d a = (v1 - v0) / dt;

    return {(v0 + v1) / 2.0, a};
  }

  const std::map<BodyId, PlanetState>& bodies() const { return bodies_; }

 private:
  // Stored as: {position_name, rotation_name, radii_name}
  static constexpr const char* BodyNames[][3] = {
      {"EARTH", "IAU_EARTH", "EARTH"},
      {"MOON", "IAU_MOON", "MOON"},
      {"SUN", "IAU_SUN", "SUN"},
      {"MERCURY BARYCENTER", "IAU_MERCURY", "MERCURY"},
      {"VENUS BARYCENTER", "IAU_VENUS", "VENUS"},
      {"MARS BARYCENTER", "IAU_MARS", "MARS"},
      {"JUPITER BARYCENTER", "IAU_JUPITER", "JUPITER"},
      {"SATURN BARYCENTER", "IAU_SATURN", "SATURN"},
      {"URANUS BARYCENTER", "IAU_URANUS", "URANUS"},
      {"NEPTUNE BARYCENTER", "IAU_NEPTUNE", "NEPTUNE"},
  };

  Sophus::SE3d T_J2000_body(BodyId id) const {
    return T_J2000_body(id, ephemerisTime_);
  }

  Sophus::SE3d T_J2000_body(BodyId id, double et) const {
    const auto index = static_cast<int>(id);
    return Sophus::SE3d(
        SpiceHelper::R_J2000_body(BodyNames[index][1], et),
        SpiceHelper::position_J2000(BodyNames[index][0], et));
  }

  double ephemerisTime_;
  BodyId origin_;
  Sophus::SE3d T_origin_J2000;
  std::map<BodyId, PlanetState> bodies_;
};

class GravitySolarSystemSim {
 public:
  GravitySolarSystemSim(const SolarSystemState& state) : state_(state) {}

  void initialize() {
    bodyStates_.clear();
    for (const auto& [body, _] : state_.bodies()) {
      const auto [velocity, acceleration] =
          state_.linearVelocityAcceleration(body, 0, 0, 0);
      bodyStates_[body] = {
          .id = body,
          .T_origin_self = state_.T_origin_body(body),
          .gMass = state_.gMass(body),
          .velocity = velocity,
          .acceleration = acceleration};
    }
    currentTimeSecs_ = state_.getTime();
    origin_ = state_.origin();
  }

  void update() {
    const double dt = state_.getTime() - currentTimeSecs_;
    if (dt == 0) {
      return;
    }

    // Update to time
    setAccelerations();
    for (auto& [_, body] : bodyStates_) {
      // Simply retrieve orientation from the state
      body.T_origin_self.so3() = state_.T_origin_body(body.id).so3();

      // Perform Euler integration forward
      body.T_origin_self.translation() += body.velocity * dt;
      body.velocity += body.acceleration * dt;
    }

    currentTimeSecs_ = state_.getTime();
  }

  BodyId origin() const { return origin_; }

  double getTime() const { return currentTimeSecs_; }

  const Sophus::SE3d& T_origin_body(BodyId id) const {
    return bodyStates_.at(id).T_origin_self;
  }

  void setOrigin(BodyId id) {
    const Sophus::SE3d T_new_oldOrigin = state_.T_origin_body(id).inverse();
    for (auto& [_, body] : bodyStates_) {
      body.T_origin_self = T_new_oldOrigin * body.T_origin_self;
    }
  }

 private:
  void setAccelerations() {
    for (auto& [_, a] : bodyStates_) {
      Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
      for (const auto& [__, b] : bodyStates_) {
        if (a.id == b.id) {
          continue;
        }
        const Eigen::Vector3d rv =
            (b.T_origin_self.translation() - a.T_origin_self.translation()) *
            1e3;
        const double r = rv.norm();
        acceleration += b.gMass / (r * r * r) * rv;
      }
      a.acceleration = acceleration / 1e3;
      fmt::println(
          "Body: {} with acceleration: {} {} {}",
          int(_),
          acceleration(0),
          acceleration(1),
          acceleration(2));
    }
  }

  const std::vector<BodyId> bodies = {
      SUN, MOON, EARTH, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE};

  struct BodyState {
    BodyId id;
    Sophus::SE3d T_origin_self;
    double gMass;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;
  };

  double currentTimeSecs_;
  BodyId origin_;
  std::map<BodyId, BodyState> bodyStates_;
  const SolarSystemState& state_;
};

class SolarSystemSimulator {
 public:
  SolarSystemSimulator() : state_(), simulator_(state_) {}

  void useSimulation() {
    if (!useSimulation_) {
      simulator_.initialize();
      useSimulation_ = true;
    }
  }

  void useSpice() { useSimulation_ = false; }

  void setTime(double newEphemerisTime) {
    state_.setTime(newEphemerisTime);
    if (useSimulation_) {
      simulator_.update();
    }
  }

  double getTime() const { return state_.getTime(); }

  void setOrigin(BodyId body) {
    if (useSimulation_) {
      simulator_.setOrigin(body);
    }
    state_.setOrigin(body);
  }

  BodyId origin() const { return state_.origin(); }

  double radius(BodyId id) const { return state_.radius(id); }

  double gMass(BodyId id) const { return state_.gMass(id); }

  Sophus::SE3d T_origin_body(BodyId id) const {
    if (useSimulation_) {
      return simulator_.T_origin_body(id);
    }
    return state_.T_origin_body(id);
  }

  Eigen::Vector3d position_body(
      BodyId id, double latitude, double longitude, double altitude) const {
    return state_.position_body(id, latitude, longitude, altitude);
  }

  std::pair<Eigen::Vector3d, Eigen::Vector3d> linearVelocityAcceleration(
      BodyId id,
      double latitude,
      double longitude,
      double altitude,
      double dt = 1,
      double scale = 1) const {
    // Never use simulation for this... just because it's simpler (and accurate)
    return state_.linearVelocityAcceleration(
        id, latitude, longitude, altitude, dt, scale);
  }

  const std::map<BodyId, SolarSystemState::PlanetState>& bodies() const {
    return state_.bodies();
  }

  double positionalErrorFromSun(BodyId id) const {
    if (useSimulation_) {
      const Sophus::SE3d gtT_sun_body =
          state_.T_origin_body(SUN).inverse() * state_.T_origin_body(id);
      const Sophus::SE3d simT_sun_body =
          simulator_.T_origin_body(SUN).inverse() *
          simulator_.T_origin_body(id);

      return (gtT_sun_body.translation() - simT_sun_body.translation()).norm();
    }
    return 0.0;
  }

 private:
  bool useSimulation_ = false;
  SolarSystemState state_;
  GravitySolarSystemSim simulator_;
};

int main() {
  // RunUnitTests();
  const std::filesystem::path parentDirectory =
      std::filesystem::path(__FILE__).parent_path();

  OpenGLWindow window("Solar System", 1980, 1080, false);
  PerPixelVAO vao;
  ReloadableShader shader(
      parentDirectory / "shaders/sdf.vert",
      parentDirectory / "shaders/single_object.frag");
  ReloadableShader dayNightShader(
      parentDirectory / "shaders/sdf.vert",
      parentDirectory / "shaders/day_night.frag");

  SolarSystemSimulator systemState;

  std::map<BodyId, ReloadableTexture> systemTextures{
      {BodyId::SUN, ReloadableTexture(parentDirectory / "assets/8k_sun.jpg")},
      {BodyId::EARTH,
       ReloadableTexture(parentDirectory / "assets/8k_earth_daymap.jpg")},
      {BodyId::MOON, ReloadableTexture(parentDirectory / "assets/moon.jpg")},
      {BodyId::MERCURY,
       ReloadableTexture(parentDirectory / "assets/mercury.jpeg")},
      {BodyId::VENUS, ReloadableTexture(parentDirectory / "assets/venus.jpeg")},
      {BodyId::MARS, ReloadableTexture(parentDirectory / "assets/mars.jpg")},
      {BodyId::JUPITER,
       ReloadableTexture(parentDirectory / "assets/jupiter.jpeg")},
      {BodyId::URANUS,
       ReloadableTexture(parentDirectory / "assets/uranus.jpeg")},
      {BodyId::SATURN,
       ReloadableTexture(parentDirectory / "assets/saturn.jpeg")},
      {BodyId::NEPTUNE,
       ReloadableTexture(parentDirectory / "assets/neptune.jpeg")}};
  ImguiOpenGLRenderer globeEarth("Globe Earth");
  ImguiOpenGLRenderer flatEarth("Azimuthal Earth");
  ImguiOpenGLRenderer equiDayNightMap("Equirectangular Map");
  ImguiOpenGLRenderer flatDayNightMap("Azimuthal Map");

  float daysPerSecond = 0; //.2;
  float cameraFieldOfViewDegs = 120.0; // 120.0f / 180.0 * M_PI;
  Eigen::Vector3f lla{56.5110 / 180.0 * M_PI, 3.5156 / 180 * M_PI, .001};
  Eigen::Vector2f pitchYaw{-35 / 180 * M_PI, 6. / 180.0 * M_PI};

  bool useSimulation = false;
  bool hideFromBody = false;
  bool isOrthographic = false;
  bool reverseTime = false;
  bool stepMode = true;
  int ymdhms[6] = {2020, 5, 7, 19, 23, 20};
  int selectedCameraMode = 0;
  int selectedFromBody = 0;
  int selectedToBody = 0;
  int flatEarthPeriods = 1;
  int selectedOriginIdx = 0;
  int selectedDayNightMapIndex = 1;
  int selectedStepDurationIndex = 0;
  bool longExposureMode = false;
  bool pause = false;
  float radiusScale = 1;
  bool showDebugObject = false;
  float debugObjectDistanceMeters = 10;
  float debugObjectHeightMeters = 5;
  float debugObjectRadiusMeters = 1;
  // systemState.setOrigin(BodyId::SUN);

  std::map<BodyId, bool> hiddenBodies;
  for (const auto& [id, _] : systemState.bodies()) {
    hiddenBodies[id] = false;
  }

  systemState.setTime(SpiceHelper::EphemerisTimeFromDate(
      ymdhms[0], ymdhms[1], ymdhms[2], ymdhms[3], ymdhms[4], ymdhms[5]));

  float flatEarthSunHeightKm = 3000;
  float flatEarthHeightVaryKm = 0;
  float flatEarthSunSizeKm = 50;

  auto previousTime = std::chrono::high_resolution_clock::now();
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  while (!window.shouldClose()) {
    window.beginNewFrame();

    // Calculate current time
    const auto currentTime = std::chrono::high_resolution_clock::now();
    auto dtSecs = (currentTime - previousTime).count() / 1e9;
    if (reverseTime) {
      dtSecs *= -1;
    }

    Sophus::SE3d T_origin_camera;

    ImGui::Begin("Scene Controls:");
    const auto dateStr =
        SpiceHelper::EphemerisTimeToDate(systemState.getTime());
    ImGui::Text("Current Date: %s", dateStr.c_str());

    ImGui::Spacing();
#if 0
    ImGui::Checkbox("Use Gravity Simulation", &useSimulation);
    if (useSimulation) {
      systemState.useSimulation();
      ImGui::Text(
          "Earth Error from Sun (km): %.2f",
          systemState.positionalErrorFromSun(BodyId::EARTH) * 1e3);
    } else {
      systemState.useSpice();
    }
#else
    systemState.useSpice();
#endif

    const auto moveTime = [&](double dt, bool override = false) {
      if (pause && !override) {
        return;
      }
      systemState.setTime(systemState.getTime() + dt);
    };

    ImGui::Spacing();
    ImGui::Text("Time Controls:");
    ImGui::Checkbox("Pause", &pause);
    ImGui::DragFloat(
        "Days per Second",
        &daysPerSecond,
        0.01,
        0,
        31,
        "%.3f",
        ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Reverse time", &reverseTime);
    ImGui::Checkbox("Start Step Mode", &stepMode);
    constexpr const char* stepDurations[] = {
        "Minute", "Hour", "Day", "Month", "Year"};
    constexpr float stepTimeSecs[] = {
        60,
        60 * 60,
        60 * 60 * 24,
        60 * 60 * 24 * int(365.25 / 12),
        60 * 60 * 60 * 24 * 365.25};
    ImGui::Combo("Step Duration", &selectedStepDurationIndex, stepDurations, 5);
    const bool forceStep = ImGui::Button("Step Forward");
    if (forceStep || stepMode) {
      moveTime(
          stepTimeSecs[selectedStepDurationIndex] * (reverseTime ? -1 : 1),
          forceStep);
    } else {
      moveTime(dtSecs * 60 * 60 * 24 * daysPerSecond * (reverseTime ? -1 : 1));
    }
    ImGui::Text("Modify Date (UTC):");
    ImGui::InputInt3("Year/Month/Day", ymdhms);
    ImGui::InputInt3("Hour/Minute/Second", ymdhms + 3);
    if (ImGui::Button("Apply")) {
      systemState.setTime(SpiceHelper::EphemerisTimeFromDate(
          ymdhms[0], ymdhms[1], ymdhms[2], ymdhms[3], ymdhms[4], ymdhms[5]));
    }

    ImGui::Separator();
    ImGui::Text("Camera Controls:");
    ImGui::Checkbox("Orthographic", &isOrthographic);
    ImGui::DragFloat(
        "Horizontal FoV",
        &cameraFieldOfViewDegs,
        1.0,
        1,
        179.0f,
        "%.0f deg",
        ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Psuedo-Long Exposure Mode", &longExposureMode);
    bool clearImages = false;
    if (longExposureMode) {
      ImGui::Checkbox("Refresh Exposure", &clearImages);
    }
    ImGui::Spacing();
    const char* cameraModes[] = {"BodyToBody", "TopDown"};
    // ImGui::Combo("Mode", &selectedCameraMode, cameraModes, 2);
    const std::string cameraMode =
        cameraModes[0]; // cameraModes[selectedCameraMode];
    static const std::map<std::string, BodyId> bodyFromString{
        {"Earth", BodyId::EARTH},
        {"Moon", BodyId::MOON},
        {"Sun", BodyId::SUN},
        {"Mercury", BodyId::MERCURY},
        {"Venus", BodyId::VENUS},
        {"Mars", BodyId::MARS},
        {"Jupiter", BodyId::JUPITER},
        {"Saturn", BodyId::SATURN},
        {"Uranus", BodyId::URANUS},
        {"Neptune", BodyId::NEPTUNE},
    };
    static const char* bodies[] = {
        "North",
        "Earth",
        "Moon",
        "Sun",
        "Mercury",
        "Venus",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune"};
    if (cameraMode == "BodyToBody") {
      ImGui::Text("Bodies:");
      ImGui::Combo("From", &selectedFromBody, bodies + 1, 10);
      ImGui::Combo("To", &selectedToBody, bodies, 11);
      // ImGui::Checkbox("Hide From?", &hideFromBody);
      const auto fromBodyId = bodyFromString.at(bodies[selectedFromBody + 1]);
      // if (hideFromBody) {
      //   hiddenBodies.insert(fromBodyId);
      // }
      ImGui::Text("Position:");
      ImGui::SliderAngle("Latitude", &lla.x(), -90, 90);
      ImGui::SliderAngle("Longitude", &lla.y(), -180, 180);
      ImGui::DragFloat("Altitude (km)", &lla.z(), 1.0, 1e-3, 1e4);
      ImGui::Text("Viewing Angle:");
      ImGui::SliderAngle("Pitch", &pitchYaw.x(), -90, 90);
      ImGui::SliderAngle("Yaw", &pitchYaw.y(), -180, 180);
      const Eigen::Vector3d position_fromBody = systemState.position_body(
          fromBodyId, lla.x(), lla.y(), lla.z() / 1e3);

      const Eigen::Vector3d up = position_fromBody.normalized();
      const Eigen::Vector3d north =
          Eigen::Vector3d::UnitZ() + -Eigen::Vector3d::UnitZ().dot(up) * up;

      if (selectedToBody == 0) {
        const Sophus::SO3d R_mount_camera =
            Sophus::SO3d::rotY(pitchYaw.y()) * Sophus::SO3d::rotX(pitchYaw.x());

        const Sophus::SE3d T_fromBody_mount =
            LookAt(position_fromBody, position_fromBody + north, up);
        T_origin_camera = systemState.T_origin_body(fromBodyId) *
            T_fromBody_mount *
            Sophus::SE3d(R_mount_camera, Eigen::Vector3d::Zero());
      } else {
        const Sophus::SO3d R_mount_camera = Sophus::SO3d::rotZ(pitchYaw.y());
        const Eigen::Vector3d mount_origin =
            systemState.T_origin_body(fromBodyId) * position_fromBody;

        const auto toBodyId = bodyFromString.at(bodies[selectedToBody]);
        const Eigen::Vector3d toBody_origin =
            systemState.T_origin_body(toBodyId).translation();

        T_origin_camera = LookAt(mount_origin, toBody_origin, up) *
            Sophus::SE3d(R_mount_camera, Eigen::Vector3d::Zero());
      }
    } else if (cameraMode == "TopDown") {
    }

    ImGui::Text("Capture Images:");
    char buffer[512];
    ImGui::InputText("Output Directory", buffer, 512, 0);
    const std::filesystem::path saveDirectory = buffer;
    if (ImGui::Button("Save Picture")) {
      if (!saveDirectory.empty() && std::filesystem::exists(saveDirectory)) {
        std::string modifiedDateStr = dateStr;
        for (char& c : modifiedDateStr)
          if (c == ':')
            c = '-';
        modifiedDateStr = fmt::format("{}.png", modifiedDateStr);
        const auto [width, height] = window.getSize();
        const GLuint textureID = globeEarth.texture();

        // Create a buffer to hold the pixel data.
        GLubyte* pixels =
            new GLubyte[width * height * 3]; // Assuming 3 channels (RGB)
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
        const std::filesystem::path filepath = saveDirectory / modifiedDateStr;
        stbi_write_png(filepath.c_str(), width, height, 3, pixels, width * 3);
      }
    }

    ImGui::Separator();
    ImGui::Text("Azimuthal Earth Parameters:");
    ImGui::SliderFloat("Sun height (km)", &flatEarthSunHeightKm, 1e-3, 10000);
    ImGui::SliderFloat("Sun size (km)", &flatEarthSunSizeKm, 1e-3, 100);
    ImGui::SliderFloat(
        "Sun height variation (km)", &flatEarthHeightVaryKm, 0, 10000);
    ImGui::SliderInt("Sun height periods", &flatEarthPeriods, 0, 10);

    ImGui::Separator();
    ImGui::Text("Statistics:");
    // ImGui::Combo("Origin:", &selectedOriginIdx, bodies + 1, 10);
    // systemState.setOrigin(bodyFromString.at(bodies[selectedOriginIdx +
    // 1]));
    const auto [velocity, acceleration] =
        systemState.linearVelocityAcceleration(
            bodyFromString.at(bodies[selectedToBody + 1]),
            lla.x(),
            lla.y(),
            lla.z() / 1e3,
            1,
            1e6);
    ImGui::Text("Camera Velocity on Earth: %.1f m/s", velocity.norm());
    ImGui::Text(
        "Camera Acceleration on Earth: %.4f m/s^2", acceleration.norm());
    ImGui::Text("%.2f%% of Gravity", acceleration.norm() / 9.81 * 100);

    ImGui::Separator();
    constexpr const char* dayNightTypes[] = {"Equirectangular", "Azimuthal"};
    ImGui::Combo("Day-Night Map", &selectedDayNightMapIndex, dayNightTypes, 2);

    ImGui::Separator();
    ImGui::Text("Debug");
    ImGui::Checkbox("Debug Object", &showDebugObject);
    ImGui::DragFloat(
        "Debug Object Height (m)", &debugObjectHeightMeters, 1, 0, 1e5);
    ImGui::DragFloat(
        "Debug Object Radius (m)", &debugObjectRadiusMeters, 1, 0, 1e5);
    ImGui::DragFloat(
        "Debug Object Distance (m)", &debugObjectDistanceMeters, 1, 0, 1e5);

    ImGui::DragFloat("Radius Scale", &radiusScale, 1.0, 0.0, 1e4);
    for (auto& [id, hidden] : hiddenBodies) {
      const std::string label = "Hide " + toString(id);
      ImGui::Checkbox(label.c_str(), &hidden);
    }

    ImGui::End();
    previousTime = currentTime;

    globeEarth.bind();
    if (!longExposureMode || clearImages) {
      globeEarth.clear();
    }
    const ImVec2 flatSize = globeEarth.size();
    const auto [flatWidth, flatHeight] = flatSize;

    const auto flatFx =
        (flatWidth / 2.0) / tan(cameraFieldOfViewDegs / 180 * M_PI / 2.0);
    Eigen::Matrix3f K;
    K << flatFx, 0, flatWidth / 2.0, 0, flatFx, flatHeight / 2.0, 0, 0, 1;

    Camera camera{
        .T_world_self = T_origin_camera,
        .K = K,
        .resolution = {flatWidth, flatHeight},
        .orthographic = isOrthographic};

    Light lighting{
        .T_self_world = systemState.T_origin_body(BodyId::SUN).inverse()};

    const auto createObject = [&](BodyId id, bool isMatte = false) {
      return SDFObject{
          .isMatte = isMatte,
          .parameters = {float(systemState.radius(id))},
          .T_self_world = systemState.T_origin_body(id).inverse(),
          .texture = systemTextures.at(id).id(),
          .type = 1};
    };

    glUseProgram(shader.id());
    glBindVertexArray(vao.id());
    for (const auto& [bodyId, _] : systemState.bodies()) {
      if (hiddenBodies.at(bodyId)) {
        continue;
      }

      auto obj = createObject(bodyId, bodyId == BodyId::SUN);
      obj.parameters.at(0) *= radiusScale;
      SetObjectUniforms(shader.id(), camera, lighting, obj);
      glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    if (showDebugObject) {
      const Sophus::SE3d T_origin_earth = systemState.T_origin_body(EARTH);
      const double percentCircumference = (debugObjectDistanceMeters / 1e6) /
          (2 * systemState.radius(EARTH) * M_PI);
      const Eigen::Vector3d debugObject_earth =
          systemState.position_body(EARTH, lla.x(), lla.y(), 0);
      const Sophus::SE3d R_obj_obj = Sophus::SE3d(
          Eigen::Quaterniond::FromTwoVectors(
              Eigen::Vector3d::UnitZ(), debugObject_earth.normalized()),
          Eigen::Vector3d::Zero());
      const Sophus::SO3d R_earth_obj =
          Sophus::SO3d::rotY(-percentCircumference * 2 * M_PI);
      const Sophus::SE3d T_origin_obj = T_origin_earth *
          Sophus::SE3d(R_earth_obj * R_obj_obj.so3(),
                       R_earth_obj * debugObject_earth);

      static GLuint defaultTextureId = CreateTexture(255, 0, 0);
      SDFObject debugObject{
          .type = 2,
          .T_self_world = T_origin_obj.inverse(),
          .parameters =
              {float(debugObjectRadiusMeters / 1e6),
               float(debugObjectHeightMeters / 1e6)},
          .texture = defaultTextureId,
          .isMatte = true};
      SetObjectUniforms(shader.id(), camera, lighting, debugObject);
      glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    globeEarth.unbind();

    flatEarth.bind();
    if (!longExposureMode || clearImages) {
      flatEarth.clear();
    }
    const auto shaderProgram = shader.id();
    glUseProgram(shaderProgram);
    glBindVertexArray(vao.id());

    const ImVec2 size = flatEarth.size();
    const auto [width, height] = size;

    const auto fx =
        (width / 2.0) / tan(cameraFieldOfViewDegs / 180 * M_PI / 2.0);
    Eigen::Matrix3f flatK;
    flatK << fx, 0, width / 2.0, 0, fx, height / 2.0, 0, 0, 1;

    SDFObject flatEarthObj{
        .type = 2,
        .T_self_world = Sophus::SE3d(),
        .parameters =
            {float(systemState.radius(BodyId::EARTH) * M_PI / 2) * radiusScale,
             .1f},
        .texture = systemTextures.at(BodyId::EARTH).id(),
        .isMatte = false,
    };

    const Sophus::SE3d T_world_surface(
        Sophus::SO3d(),
        Eigen::Vector3d::UnitZ() * flatEarthObj.parameters.at(1));

    const Sophus::SO3d R_mount_camera =
        Sophus::SO3d::rotY(pitchYaw.y()) * Sophus::SO3d::rotX(pitchYaw.x());

    const double distanceFromCenter =
        ((-lla.x() + M_PI / 2) / M_PI) * flatEarthObj.parameters.at(0);
    Eigen::Vector3d mountPosition_surface;
    mountPosition_surface.x() =
        distanceFromCenter * std::cos(lla.y() + M_PI / 2);
    mountPosition_surface.y() =
        distanceFromCenter * std::sin(lla.y() + M_PI / 2);
    mountPosition_surface.z() = mountPosition_surface.z() = lla.z() / 1e3;
    Sophus::SE3d T_surface_mount = LookAt(
        mountPosition_surface,
        Eigen::Vector3d::UnitZ() * mountPosition_surface.z(),
        Eigen::Vector3d::UnitZ());

    Camera flatCamera{
        .T_world_self = T_world_surface * T_surface_mount *
            Sophus::SE3d(R_mount_camera, Eigen::Vector3d::Zero()),
        .resolution = {width, height},
        .K = flatK,
        .orthographic = isOrthographic};

    const double angle = M_PI * .4 +
        -(std::fmod(systemState.getTime(), 60 * 60 * 24) / (60 * 60 * 24)) *
            (2 * M_PI);
    const double percentYear =
        std::fmod(systemState.getTime(), 60 * 60 * 24 * 365.25) /
        (60 * 60 * 24 * 365.25);
    const double sunLatitude =
        std::sin(percentYear * 2 * M_PI) * 23.5 / 180.0 * M_PI;
    const double sunDistanceFromCenter =
        ((-sunLatitude + M_PI / 2) / M_PI) * flatEarthObj.parameters.at(0);
    Eigen::Vector3d sunPosition_surface;
    sunPosition_surface.x() = sunDistanceFromCenter * std::cos(angle);
    sunPosition_surface.y() = sunDistanceFromCenter * std::sin(angle);
    sunPosition_surface.z() = flatEarthSunHeightKm / 1e3 +
        (std::sin(percentYear * 2 * M_PI * flatEarthPeriods) *
         flatEarthHeightVaryKm / 1e3);

    const double spinAngle =
        (std::fmod(systemState.getTime(), 60 * 60 * 24) / (60 * 60 * 24 * 27)) *
        (2 * M_PI);

    Sophus::SE3d T_surface_sun(
        Sophus::SO3d::rotZ(spinAngle), sunPosition_surface);

    SDFObject sunObject{
        .type = 1,
        .T_self_world = (T_world_surface * T_surface_sun).inverse(),
        .parameters = {float(flatEarthSunSizeKm / 1e3) * radiusScale},
        .texture = systemTextures.at(BodyId::SUN).id(),
        .isMatte = true};

    Light flatLighting{.T_self_world = sunObject.T_self_world};
    SetObjectUniforms(shaderProgram, flatCamera, flatLighting, flatEarthObj);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    SetObjectUniforms(shaderProgram, flatCamera, flatLighting, sunObject);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    flatEarth.unbind();

    int mapType = 0;
    for (auto pDayNightMap : std::vector<ImguiOpenGLRenderer*>{
             &equiDayNightMap, &flatDayNightMap}) {
      auto& dayNightMap = *pDayNightMap;
      dayNightMap.bind();
      dayNightMap.clear();
      glUseProgram(dayNightShader.id());
      glBindVertexArray(vao.id());
      const auto sid = dayNightShader.id();

      const Eigen::Vector3f lightPosition_sphere =
          (systemState.T_origin_body(EARTH).inverse() *
           systemState.T_origin_body(SUN).translation())
              .cast<float>();
      glUniform3fv(
          glGetUniformLocation(sid, "lightPosition_sphere"),
          1,
          lightPosition_sphere.data());
      glUniform1f(
          glGetUniformLocation(sid, "sphereRadius"), systemState.radius(EARTH));
      const Eigen::Vector2f dnResolution{
          dayNightMap.size().x, dayNightMap.size().y};
      glUniform2fv(
          glGetUniformLocation(sid, "resolution"), 1, dnResolution.data());
      glUniform1i(glGetUniformLocation(sid, "mapType"), mapType++);
      glBindTexture(GL_TEXTURE_2D, systemTextures.at(EARTH).id());
      glDrawArrays(GL_TRIANGLES, 0, 6);

      dayNightMap.unbind();
    }

    window.finishFrame();
  }

  // Clean up textures to be nice
  for (auto& [_, texture] : systemTextures) {
    texture.release();
  }

  return 0;
}
