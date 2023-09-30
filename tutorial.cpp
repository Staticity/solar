#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include "imgui/imgui.h"

#include <iostream>
// #include <glad/glad.h>
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Vertex Shader source code
const char* vertexShaderSource =
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "uniform float size;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(size * aPos.x, size * aPos.y, size * aPos.z, 1.0);\n"
    "}\0";
// Fragment Shader source code
const char* fragmentShaderSource =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "uniform vec4 color;\n"
    "void main()\n"
    "{\n"
    "   FragColor = color;\n"
    "}\n\0";

int main() {
  // Initialize GLFW
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Create a GLFWwindow object of 800 by 800 pixels, naming it "YoutubeOpenGL"
  GLFWwindow* window = glfwCreateWindow(800, 800, "ImGui + GLFW", NULL, NULL);
  // Error check if the window fails to create
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }

  // Introduce the window into the current context
  glfwMakeContextCurrent(window);

  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    return -1;
  }

  // Load GLAD so it configures OpenGL
  //   gladLoadGL();
  // Specify the viewport of OpenGL in the Window
  // In this case the viewport goes from x = 0, y = 0, to x = 800, y = 800
  glViewport(0, 0, 800, 800);

  // Create Vertex Shader Object and get its reference
  std::cout << "test" << std::endl;
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  std::cout << "test" << std::endl;
  // Attach Vertex Shader source to the Vertex Shader Object
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  // Compile the Vertex Shader into machine code
  glCompileShader(vertexShader);

  // Create Fragment Shader Object and get its reference
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  // Attach Fragment Shader source to the Fragment Shader Object
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  // Compile the Vertex Shader into machine code
  glCompileShader(fragmentShader);

  // Create Shader Program Object and get its reference
  GLuint shaderProgram = glCreateProgram();
  // Attach the Vertex and Fragment Shaders to the Shader Program
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  // Wrap-up/Link all the shaders together into the Shader Program
  glLinkProgram(shaderProgram);

  // Delete the now useless Vertex and Fragment Shader objects
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  // Vertices coordinates
  GLfloat vertices[] = {
      -0.5f,
      -0.5f * float(std::sqrt(3)) / 3,
      0.0f, // Lower left corner
      0.5f,
      -0.5f * float(std::sqrt(3)) / 3,
      0.0f, // Lower right corner
      0.0f,
      0.5f * float(std::sqrt(3)) * 2 / 3,
      0.0f // Upper corner
  };

  // Create reference containers for the Vartex Array Object and the Vertex
  // Buffer Object
  GLuint VAO, VBO;

  // Generate the VAO and VBO with only 1 object each
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);

  // Make the VAO the current Vertex Array Object by binding it
  glBindVertexArray(VAO);

  // Bind the VBO specifying it's a GL_ARRAY_BUFFER
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  // Introduce the vertices into the VBO
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  // Configure the Vertex Attribute so that OpenGL knows how to read the VBO
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  // Enable the Vertex Attribute so that OpenGL knows to use it
  glEnableVertexAttribArray(0);

  // Bind both the VBO and VAO to 0 so that we don't accidentally modify the VAO
  // and VBO we created
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

  // Variables to be changed in the ImGUI window
  bool drawTriangle = true;
  float size = 1.0f;
  float color[4] = {0.8f, 0.3f, 0.02f, 1.0f};

  // Exporting variables to shaders
  glUseProgram(shaderProgram);
  glUniform1f(glGetUniformLocation(shaderProgram, "size"), size);
  glUniform4f(
      glGetUniformLocation(shaderProgram, "color"),
      color[0],
      color[1],
      color[2],
      color[3]);

  // Main while loop
  while (!glfwWindowShouldClose(window)) {
    // Tell OpenGL a new frame is about to begin
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("OpenGL");

    ImVec2 pos = ImGui::GetWindowPos();
    ImVec2 s = ImGui::GetWindowSize();
    glViewport((int)pos.x, (int)(io.DisplaySize.y - pos.y), (int)s.x, (int)s.y);
    glEnable(GL_SCISSOR_TEST);
    glScissor(
        (int)pos.x, (int)(io.DisplaySize.y - pos.y - s.y), (int)s.x, (int)s.y);

    // Your OpenGL drawing commands...

    // Specify the color of the background
    glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
    // Clean the back buffer and assign the new color to it
    glClear(GL_COLOR_BUFFER_BIT);

    // Tell OpenGL which Shader Program we want to use
    glUseProgram(shaderProgram);
    // Bind the VAO so OpenGL knows to use it
    glBindVertexArray(VAO);
    // Only draw the triangle if the ImGUI checkbox is ticked
    if (drawTriangle)
      // Draw the triangle using the GL_TRIANGLES primitive
      glDrawArrays(GL_TRIANGLES, 0, 3);

    glDisable(GL_SCISSOR_TEST);
    ImGui::End();

    // ImGUI window creation
    ImGui::Begin("My name is window, ImGUI window");
    // Text that appears in the window
    ImGui::Text("Hello there adventurer!");
    // Checkbox that appears in the window
    ImGui::Checkbox("Draw Triangle", &drawTriangle);
    // Slider that appears in the window
    ImGui::SliderFloat("Size", &size, 0.5f, 2.0f);
    // Fancy color editor that appears in the window
    ImGui::ColorEdit4("Color", color);
    // Ends the window
    ImGui::End();
    ImGui::Begin("DockSpace Demo"); // Create a new window
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id);
    ImGui::End();
    // Export variables to shader
    glUseProgram(shaderProgram);
    glUniform1f(glGetUniformLocation(shaderProgram, "size"), size);
    glUniform4f(
        glGetUniformLocation(shaderProgram, "color"),
        color[0],
        color[1],
        color[2],
        color[3]);

    // Renders the ImGUI elements
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Swap the back buffer with the front buffer
    glfwSwapBuffers(window);
    // Take care of all GLFW events
    glfwPollEvents();
  }

  // Deletes all ImGUI instances
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  // Delete all the objects we've created
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteProgram(shaderProgram);
  // Delete window before ending the program
  glfwDestroyWindow(window);
  // Terminate GLFW before ending the program
  glfwTerminate();
  return 0;
}