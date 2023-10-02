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

  ~ReloadableTexture() { glDeleteTextures(1, &texture_); }

  ReloadableTexture(const ReloadableTexture&) = delete;
  ReloadableTexture& operator=(const ReloadableTexture&) = delete;

  ReloadableTexture(ReloadableTexture&& other) noexcept
      : texture_(other.texture_), filepath_(std::move(other.filepath_)) {
    other.texture_ = 0; // Transfer ownership
  }

  ReloadableTexture& operator=(ReloadableTexture&& other) noexcept {
    if (this != &other) {
      glDeleteTextures(1, &texture_); // Clean up existing texture
      texture_ = other.texture_;
      filepath_ = std::move(other.filepath_);
      other.texture_ = 0; // Transfer ownership
    }
    return *this;
  }

  void load(const std::filesystem::path& path) {
    filepath_ = FileModifiedTracker(path);
    texture_ = CreateTexture(filepath_->path());
  }

  void maybeReload() const {
    if (filepath_ && filepath_->wasModified()) {
      texture_ = CreateTexture(filepath_->path());
    }
  }

  GLuint id() const {
    maybeReload();
    return texture_;
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

class SPICEHelper {
 public:
  static SPICEHelper& getInstance() {
    static SPICEHelper instance;
    return instance;
  }

  static Sophus::SE3d T_J2000_body(
      const std::string& body, double ephemerisTime) {
    return getInstance().T_J2000_body_impl(body, ephemerisTime);
  }

  static SpiceDouble EphemerisTimeNow() {
    // Ensure the instance is instantitaed for kernels
    (void)getInstance();

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

  // struct Date {
  //   int year;
  //   int month;
  //   int day;
  //   int hour;
  //   int minute;
  //   int second;
  // };
  static std::string EphemerisTimeToDate(SpiceDouble et) {
    char utc_time[40];
    et2utc_c(et, "%Y-%m-%dT%H:%M:%S", 0, 40, utc_time);
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

 private:
  SPICEHelper() { init(); }
  ~SPICEHelper() { cleanup(); }

  // Non-static implementation for T_J2000_body, which will be called by the
  // static wrapper
  Sophus::SE3d T_J2000_body_impl(
      const std::string& body, double ephemerisTime) {
    fmt::println("SPICE Position");

    SpiceDouble bodyState[6], lt;
    spkezr_c(
        body.c_str(), ephemerisTime, "J2000", "NONE", "EARTH", bodyState, &lt);

    fmt::println("SPICE Orientation");
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

  const std::vector<std::string> kernelPaths_ = {
      "/Users/static/Downloads/cspice/data/naif0012.tls",
      "/Users/static/Downloads/cspice/data/de440.bsp",
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

  // Delete copy and move constructors and assign operators
  SPICEHelper(SPICEHelper const&) = delete; // Copy constructor
  SPICEHelper(SPICEHelper&&) = delete; // Move constructor
  SPICEHelper& operator=(SPICEHelper const&) = delete; // Copy assignment
  SPICEHelper& operator=(SPICEHelper&&) = delete; // Move assignment
};

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
  ReloadableTexture texture;

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

  glBindTexture(GL_TEXTURE_2D, object.texture.id());
}

int main() {
  OpenGLWindow window("Solar System", 1980, 1080, true);
  PerPixelVAO vao;

  ReloadableShader shader(
      "/Users/static/Documents/code/sdfs/shaders/sdf.vert",
      "/Users/static/Documents/code/sdfs/shaders/single_object.frag");

  // Get the heliocentric position of Earth (which gives us the position of the
  // Sun relative to Earth)

  SDFObject sun{
      .type = 1,
      .T_self_world = {},
      .parameters = {696.34},
      .texture = ReloadableTexture(
          "/Users/static/Documents/code/sdfs/assets/8k_sun.jpg"),
      .isMatte = true};
  SDFObject earth{
      .type = 1,
      .T_self_world = {},
      .parameters = {6.371009},
      .texture = ReloadableTexture(
          "/Users/static/Documents/code/sdfs/assets/earth.jpg"),
      .isMatte = false};
  SDFObject moon{
      .type = 1,
      .T_self_world = {},
      .parameters = {1.7374},
      .texture = ReloadableTexture(
          "/Users/static/Documents/code/sdfs/assets/moon2.jpg"),
      .isMatte = false};

  float daysPerSecond = 0; // .1;
  float cameraFieldOfView = 1.0f / 180.0 * M_PI;
  Eigen::Vector3f lla{39.2 / 180 * M_PI, -1.18 / 180 * M_PI, 3};
  // Eigen::Vector3f lla{47.608013 / 180 * M_PI, -122.335167 / 180 * M_PI, 3};

  ImguiOpenGLRenderer gameTab("Game");

  bool hideEarth = false;
  // earth, up, moon, sun
  bool lookat[4] = {false, false, false, false};
  glEnable(GL_DEPTH_TEST);

  int ymdhms[6] = {2023, 10, 13, 0, 0, 0};

  SpiceDouble currentEt = 750578468.1823622; // SPICEHelper::EphemerisTimeNow();
  auto previousTime = std::chrono::high_resolution_clock::now();
  while (!window.shouldClose()) {
    window.beginNewFrame();

    const char* lookatOptions[] = {
        "Earth", "Horizon", "Moon", "Sun", "EarthTopDown", "SunTopDown"};
    static int currentLook = 2;

    const char* originOptions[] = {"J2000", "Earth", "Moon", "Sun"};
    static int currentOrigin = 0;

    // ImGUI window creation
    ImGui::Begin("Controls");
    ImGui::Text("Time:");
    ImGui::SliderFloat("Days per Second", &daysPerSecond, 0, 31);
    ImGui::Text("Date (UTC):");
    ImGui::InputInt3("Year/Month/Day", ymdhms);
    ImGui::InputInt3("Hour/Minute/Second", ymdhms + 3);
    if (ImGui::Button("Set Date")) {
      currentEt = SPICEHelper::EphemerisTimeFromDate(
          ymdhms[0], ymdhms[1], ymdhms[2], ymdhms[3], ymdhms[4], ymdhms[5]);
    }

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
    ImGui::End();

    // Calculate current time
    const auto currentTime = std::chrono::high_resolution_clock::now();
    const auto dtSecs = (previousTime - currentTime).count() / 1e9;
    previousTime = currentTime;

    currentEt += dtSecs * 60 * 60 * 24 * daysPerSecond;
    Sophus::SE3d T_J2000_earth = SPICEHelper::T_J2000_body("EARTH", currentEt);
    Sophus::SE3d T_J2000_sun = SPICEHelper::T_J2000_body("SUN", currentEt);
    Sophus::SE3d T_J2000_moon = SPICEHelper::T_J2000_body("MOON", currentEt);
    // (void)SPICEHelper::T_J2000_body("MARS", currentEt);

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
    const auto lat = -lla.x();
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

    // Calculate viewing angle between sun and moon from camera
    // {
    //   Eigen::Vector3d ab = (T_earth_camera.inverse() * earth.T_self_world *
    //                         sun.T_self_world.inverse())
    //                            .translation();
    //   Eigen::Vector3d ac = (T_earth_camera.inverse() * earth.T_self_world *
    //                         moon.T_self_world.inverse())
    //                            .translation();
    //   // std::string d = SPICEHelper::EphemerisTimeToDate(currentEt);
    //   const double angle =
    //       180 / M_PI * std::acos((ab.dot(ac)) / (ab.norm() * ac.norm()));
    //   if (angle < 2)
    //     fmt::println("{}: {:.2f}", currentEt, angle);
    // }

    // Show line intersection of Sun-Moon line onto earth
    // std::optional<SDFObject> object;
    // {
    //   const Eigen::Vector3d center =
    //   sun.T_self_world.inverse().translation(); const Eigen::Vector3d
    //   direction =
    //       (moon.T_self_world.inverse().translation() - center).normalized();
    //   if (const auto distance = intersect(
    //           center,
    //           direction,
    //           earth.T_self_world.inverse().translation(),
    //           earth.parameters.front())) {
    //     static int xx = 0;
    //     fmt::println("{}", xx++);
    //     fmt::println("Found an eclipse!");
    //     const Eigen::Vector3d hit = center + direction * distance.value();
    //     object = SDFObject{
    //         .isMatte = true,
    //         .parameters = {earth.parameters.front() / 10},
    //         .texture = ReloadableTexture(255, 0, 0),
    //         .type = 1,
    //         .T_self_world = Sophus::SE3d(Sophus::SO3d(), hit).inverse()};
    //     fmt::println(
    //         "radius: {}, earth distance: {}",
    //         object->parameters.front(),
    //         (earth.T_self_world * object->T_self_world.inverse())
    //             .translation()
    //             .norm());
    //   }
    // }

    gameTab.bind();
    gameTab.clear();
    const ImVec2 size = gameTab.size();
    const auto [width, height] = size;

    const auto fy = (height / 2.0) / tan(cameraFieldOfView / 2.0);
    Eigen::Matrix3f K;
    K << fy, 0, width / 2.0, 0, fy, height / 2.0, 0, 0, 1;

    Camera camera{
        .T_world_self = earth.T_self_world.inverse() * T_earth_camera, .K = K};

    Light lighting{.T_self_world = sun.T_self_world};

    const auto shaderProgram = shader.id();
    glUseProgram(shaderProgram);
    glBindVertexArray(vao.id());
    SetObjectUniforms(shaderProgram, camera, lighting, sun);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    SetObjectUniforms(shaderProgram, camera, lighting, moon);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    if (!hideEarth) {
      SetObjectUniforms(shaderProgram, camera, lighting, earth);
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

  return 0;
}
