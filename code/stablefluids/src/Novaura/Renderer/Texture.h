#pragma once

namespace Novaura {

	class Texture
	{
	public:
		Texture() = default;
		Texture(const std::string& path);
		Texture(std::string_view path);

		void Bind(unsigned int slot = 0) const;
		void UnBind() const;
		//void UnBindSlot(unsigned int slot = 0) const;

		//void BindSlot(unsigned int slot = 0)const;

		inline int GetWidth() const { return m_Width; }
		inline int GetHeight() const { return m_Height; }
		inline unsigned int GetID() const { return m_TextureID; }			
		inline void SetSlot(uint32_t slot) { m_Slot = slot; }
		inline uint32_t GetSlot() const { return m_Slot; }
		//void SetID(unsigned int id) { m_TextureID = id; }		

		inline std::string_view GetTextureFile() const { return m_TextureFile; }
		inline void SetTextureFile(std::string_view textureFile) { m_TextureFile = textureFile; }
	
	private:
		unsigned int m_TextureID;
		int m_Width, m_Height, m_NumChannels;

		mutable uint32_t m_Slot;
		std::string_view m_TextureFile;
		void LoadTexture(const std::string& path);
		void LoadTexture(std::string_view path);

	};

}