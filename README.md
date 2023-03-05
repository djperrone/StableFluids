# StableFluids
![image](https://github.com/djperrone/StableFluids/blob/main/title_img/title_image_cropped.jpg)
Implementing Jos Stam's stable fluids algorithm in parallel on Cuda and rendering it using Cuda-OpenGL interop for our final project in CSC 415 (parallel computing)
#### By David Perrone and Jake Afonso

### Project Structure
- Rendering, camera, opengl context and input are handled in the novaura folder
- Fluid Simulation Code is in the App folder
- App/States/StableFluidsGPU.cpp is the main source file for the demo and handles updates and draw calls
- gui.cpp is the user interface (made with imgui)
- Novaura/Renderer/Renderer.cpp handles cuda opengl interop and instanced rendering
- The CudaSrc folder contains all cuda source code
  - Fluid.cu(h) and utilities.cu(h) is where the simulation calculations happen
  - CudaMath.cu(h) is where the matrix multiplication and color updates happen

### How to use the demo
- arrow keys to move camera
- scroll wheel to zoom in and out
- gui window is used to control color, velocity, and density of dye and background

### PC Requirements
- Windows 10
- Nvidia Graphics Card (tested with gtx 960 and 970 - results may vary on lower graphics cards)
- Nvidia Cuda Toolkit 12.1
- Visual Studio 2019

### How to run
- git clone (must use git command line for submodules to work)
- git submodule update --init --recursive
- Run the GenerateProjects batch file located in Code folder
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
- Interop source: https://www.nvidia.com/content/GTC/documents/1055_GTC09.pdf
- Jos Stam: https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
- Mike Ash: https://mikeash.com/pyblog/fluid-simulation-for-dummies.html
