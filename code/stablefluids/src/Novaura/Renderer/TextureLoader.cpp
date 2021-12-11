#include "sapch.h"
#include "TextureLoader.h"
#include <spdlog/spdlog.h>

namespace Novaura {

    std::unordered_map<std::string, Texture> TextureLoader::LoadedTextures;

   /* Novaura::Texture TextureLoader::LoadTexture(const std::string& path)
    {
        spdlog::trace("loadtexture1");
        if (LoadedTextures.find(path) != LoadedTextures.end())
        {
            return LoadedTextures[path];
        }
        else
        {
            LoadedTextures[path] = Novaura::Texture(path);
            return LoadedTextures[path];
        }
    }*/

    Texture TextureLoader::LoadTexture(std::string_view path)
    {
        spdlog::trace("loadtexture1");
        if (LoadedTextures.find(path.data()) != LoadedTextures.end())
        {
            return LoadedTextures[path.data()];
        }
        else
        {
            LoadedTextures[path.data()] = Texture(path);
            return LoadedTextures[path.data()];
        }
    }

    void TextureLoader::CacheTexture(std::string_view path)
    {
        if (LoadedTextures.find(path.data()) == LoadedTextures.end())
        {
            LoadedTextures[path.data()] = Texture(path);
        }       
    }

    // batch renderer
    std::unordered_map <std::string_view, std::shared_ptr<Texture>> BatchTextureLoader::LoadedTextures;

    std::shared_ptr<Texture> BatchTextureLoader::LoadTexture(std::string_view path)
    {
        if (LoadedTextures.find(path.data()) != LoadedTextures.end())
        {
            return LoadedTextures[path.data()];
        }
        else
        {
            LoadedTextures[path.data()] = std::make_shared<Texture>(path);
        }
        return LoadedTextures[path.data()];
    }

    void BatchTextureLoader::CacheTexture(std::string_view path)
    {
        if (LoadedTextures.find(path.data()) == LoadedTextures.end())
        {
            LoadedTextures[path.data()] = std::make_shared<Texture>(path);
        }
    }
  

}
