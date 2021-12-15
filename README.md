# StableFluids

Implementing Jos Stam's stable fluids algorithm in parallel on Cuda and rendering it using Cuda-OpenGL interop


### PC Requirements
- Windows 10
- Nvidia Graphics Card
- Nvidia Cuda Toolkit
- Visual Studio 2019

### Project Structure
- Rendering, camera, opengl context and input are handled in the novaura folder
- Fluid Simulation Code is in the App folder
- App/States/StableFluidsGPU.cpp is the main source file for the demo
- Novaura/Renderer/Renderer.cpp handles cuda opengl interop and instanced rendering

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
- Jos Stam: https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
- Mike Ash: https://mikeash.com/pyblog/fluid-simulation-for-dummies.html
