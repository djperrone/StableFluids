# StableFluids

### PC Requirements
- Windows 10
- Nvidia Graphics Card
- Nvidia Cuda Toolkit
- Visual Studio 2019

### How to use the demo
- arrow keys to move camera
- scroll wheel to zoom in and out
- gui window is used to control color, velocity, and density of dye and background

### How to run
- git clone
- git submodule update --init --recursive
- Run the GenerateProjects batch file
- Open the project and manually include cu and cuh files in the project
- F5 to run (Release mode recommended)

### Dependencies
- glfw
- glad
- imgui
- spdlog

### Sources
- Project structure based on Cherno's OpenGL and Hazel series
- Rendering code based on LearnOpenGL.com's instanced rendering chapter
