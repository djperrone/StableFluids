workspace "StableFluids"
	architecture "x64"
	startproject "StableFluids"

	configurations
	{
		"Debug",
		"Release"
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

-- include directories relative to root folder (solution directory)

IncludeDir = {}
IncludeDir["GLFW"] = "StableFluids/vendor/GLFW/include"
IncludeDir["Glad"] = "StableFluids/vendor/Glad/include"
IncludeDir["ImGui"] = "StableFluids/vendor/imgui"
IncludeDir["stb_image"] = "StableFluids/vendor/stb_image"
IncludeDir["glm"] = "StableFluids/vendor/glm"


group "Dependencies"
include "StableFluids/vendor/GLFW"
include "StableFluids/vendor/Glad"
include "StableFluids/vendor/imgui"
group ""

project "StableFluids"
	location "StableFluids"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "on"
	buildcustomizations "BuildCustomizations/CUDA 12.1"
	

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")	

	pchheader "sapch.h"
	pchsource "StableFluids/src/sapch.cpp"

	defines
	{
		"_CRT_SECURE_NO_WARNINGS"	
	}

	files
	{
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp",
		"%{prj.name}/vendor/stb_image/**.h",
		"%{prj.name}/vendor/stb_image/**.cpp",	
		"%{prj.name}/vendor/glm/glm/**.hpp",
		"%{prj.name}/vendor/glm/glm/**.inl"
	}

	

	includedirs
	{
		"%{prj.name}/src",		
		"%{prj.name}/vendor/spdlog/include",						
		"%{IncludeDir.GLFW}",
		"%{IncludeDir.Glad}",
		"%{IncludeDir.ImGui}",	
		"%{IncludeDir.stb_image}",	
		"%{IncludeDir.glm}"
	}
	libdirs
	{
		--"%{prj.name}/vendor/imgui/bin"
	}

	links
	{
		"GLFW",
		"Glad",				
		 "ImGui",		
		"opengl32.lib"
	}

	filter "system:windows"
		systemversion "latest"

		defines
		{			
			"GLFW_INCLUDE_NONE"		
		}

	filter "configurations:Debug"			
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"
