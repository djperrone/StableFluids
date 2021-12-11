#pragma once

#include "Novaura/Renderer/Texture.h"

namespace Novaura {

    class TextureLoader
    {
    public:
       // static Texture LoadTexture(const std::string& path);
        static Texture LoadTexture(std::string_view path);
        static void CacheTexture(std::string_view path);
        static std::unordered_map<std::string, Texture> LoadedTextures;
    };

    class BatchTextureLoader
    {
    public:
        static std::shared_ptr<Texture> LoadTexture(const std::string& path);
        static std::shared_ptr<Texture> LoadTexture(std::string_view path);
        static void CacheTexture(std::string_view path);
        static std::unordered_map<std::string_view, std::shared_ptr<Texture>> LoadedTextures;
    };

   
}