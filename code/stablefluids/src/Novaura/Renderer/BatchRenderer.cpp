#include "sapch.h"
#include "BatchRenderer.h"

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include "Novaura/Renderer/TextureLoader.h"

#include "Novaura/Renderer/Vertex.h"
#include "Novaura/Renderer/IndexBuffer.h"
#include "Novaura/Renderer/VertexArray.h"
#include "Novaura/Renderer/VertexBuffer.h"


// store main shape then store outline shape
// draw call - statr 0 inc by 2 - actual render fill stencil buffer
// after start 1, inc 2 - stencil draw

namespace Novaura {

	struct BatchRendererData
	{
		static const uint32_t MaxQuads = 10'000;
		static const uint32_t MaxVertices = MaxQuads * 4;
		static const uint32_t MaxIndices = MaxQuads * 6;
		uint32_t QuadIndexCount = 0;

		std::unique_ptr<VertexArray> BatchVertexArray;
		std::unique_ptr<IndexBuffer> BatchIndexBuffer;
		std::unique_ptr<VertexBuffer> BatchVertexBuffer;

		
		uint32_t StencilIndex = 0;

		std::unique_ptr<Shader> TextureShader;
		std::unique_ptr<Shader> TextRenderShader;
		std::unique_ptr<Shader> CircleShader;

		glm::vec4 DefaultRectangleVertices[4];
		glm::vec2 DefaultTextureCoords[4];
		glm::vec2 NegativeTextureCoords = glm::vec2(-1.0f,-1.0f);
		

		std::vector<VertexData> BatchRectBuffer;
		std::vector<VertexData> StencilRectBuffer;
		std::vector<VertexData> CircleBuffer;
		
		static const uint32_t MaxTextures = 32;
		uint32_t CurrentTextureSlot = 1;
		std::array<std::shared_ptr<Texture>, MaxTextures> Textures;	
	};

	static BatchRendererData s_RenderData;

	void BatchRenderer::Init()
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);
		SetClearColor(0.05f, 0.05f, 0.05f, 1.0f);

		//--------------------------------------------------------------------
		s_RenderData.BatchVertexArray = std::make_unique<VertexArray>();
		s_RenderData.BatchVertexBuffer = std::make_unique<VertexBuffer>();
		s_RenderData.BatchVertexArray->Bind();
		s_RenderData.BatchVertexBuffer->Bind();
		s_RenderData.TextureShader = std::make_unique<Shader>("Assets/Shaders/BatchTextureShader.glsl");
		s_RenderData.TextRenderShader = std::make_unique<Shader>("Assets/Shaders/TextRenderShader.glsl");
		s_RenderData.CircleShader = std::make_unique<Shader>("Assets/Shaders/CircleShader.glsl");

		s_RenderData.BatchVertexArray->AddBuffer(*s_RenderData.BatchVertexBuffer, 0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), 0);
		s_RenderData.BatchVertexArray->AddBuffer(*s_RenderData.BatchVertexBuffer, 1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, Color));
		s_RenderData.BatchVertexArray->AddBuffer(*s_RenderData.BatchVertexBuffer, 2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, TexCoord));
		s_RenderData.BatchVertexArray->AddBuffer(*s_RenderData.BatchVertexBuffer, 3, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, Quantity));
		s_RenderData.BatchVertexArray->AddBuffer(*s_RenderData.BatchVertexBuffer, 4, 1, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, TextureSlot));
	
		
		s_RenderData.DefaultRectangleVertices[0] = glm::vec4(-0.5f, -0.5f, 0.0f, 1.0f);
		s_RenderData.DefaultRectangleVertices[1] = glm::vec4(0.5f, -0.5f, 0.0f, 1.0f);
		s_RenderData.DefaultRectangleVertices[2] = glm::vec4(0.5f, 0.5f, 0.0f, 1.0f);
		s_RenderData.DefaultRectangleVertices[3] = glm::vec4(-0.5f, 0.5f, 0.0f, 1.0f);

		s_RenderData.DefaultTextureCoords[0] = glm::vec2(0.0f, 0.0f);
		s_RenderData.DefaultTextureCoords[1] = glm::vec2(1.0f, 0.0f);
		s_RenderData.DefaultTextureCoords[2] = glm::vec2(1.0f, 1.0f);
		s_RenderData.DefaultTextureCoords[3] = glm::vec2(0.0f, 1.0f);



		std::vector<uint32_t> indices;
		indices.reserve(s_RenderData.MaxIndices);

		uint32_t offset = 0;
		for (auto i = 0; i < s_RenderData.MaxIndices; i += 6)
		{
			indices.emplace_back(0 + offset);
			indices.emplace_back(1 + offset);
			indices.emplace_back(2 + offset);
			indices.emplace_back(2 + offset);
			indices.emplace_back(3 + offset);
			indices.emplace_back(0 + offset);

			offset += 4;
		}		

		s_RenderData.BatchIndexBuffer = std::make_unique <IndexBuffer>(&indices[0], s_RenderData.MaxIndices);		
		s_RenderData.BatchIndexBuffer->Bind();

		int32_t samplers[s_RenderData.MaxTextures];
		for (uint32_t i = 0; i < s_RenderData.MaxTextures; i++)
		{
			samplers[i] = i;
		}

		s_RenderData.TextureShader->Bind();
		s_RenderData.TextureShader->SetIntArray("u_Textures", samplers, s_RenderData.MaxTextures);


		// stencil
		glEnable(GL_STENCIL_TEST);
		glStencilMask(0x00);
		glStencilFunc(GL_NOTEQUAL, 1, 0XFF);
		glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);		

		
	}

	void BatchRenderer::SetClearColor(float r, float g, float b, float a)
	{
		glClearColor(r, g, b, 1.0f);
	}

	void BatchRenderer::Clear()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	}

	void BatchRenderer::BeginScene(const Camera& camera)
	{		
		//s_RenderData.TextRenderShader->Bind();
		//s_RenderData.TextRenderShader->SetUniformMat4f("u_ViewProjectionMatrix", camera.GetViewProjectionMatrix());
		s_RenderData.CircleShader->Bind();

		s_RenderData.CircleShader->SetUniformMat4f("u_ViewProjectionMatrix", camera.GetViewProjectionMatrix());

		s_RenderData.TextureShader->Bind();

		s_RenderData.TextureShader->SetUniformMat4f("u_ViewProjectionMatrix", camera.GetViewProjectionMatrix());

		s_RenderData.BatchRectBuffer.reserve(s_RenderData.MaxVertices * sizeof(VertexData));
		s_RenderData.StencilRectBuffer.reserve(s_RenderData.MaxVertices * sizeof(VertexData));
		s_RenderData.CircleBuffer.reserve(s_RenderData.MaxVertices * sizeof(VertexData));
		s_RenderData.QuadIndexCount = 0;

	}
	void BatchRenderer::EndScene()
	{		
		for (uint32_t i = 1; i < s_RenderData.CurrentTextureSlot; i++)
		{
			s_RenderData.Textures[i]->Bind(i);
		}
		s_RenderData.BatchVertexArray->Bind();
		s_RenderData.BatchVertexBuffer->Bind();
		s_RenderData.BatchIndexBuffer->Bind();

		if (s_RenderData.BatchRectBuffer.size() > 0)
		{
			//s_RenderData.BatchVertexArray->Bind();
			//s_RenderData.BatchVertexBuffer->Bind();
			//s_RenderData.BatchIndexBuffer->Bind();
			s_RenderData.TextureShader->Bind();

			s_RenderData.BatchVertexBuffer->SetData(s_RenderData.BatchRectBuffer);
			glDrawElements(GL_TRIANGLES, (s_RenderData.BatchRectBuffer.size() / 4 * 6), GL_UNSIGNED_INT, nullptr);
		}
		

		s_RenderData.BatchRectBuffer.clear();
		s_RenderData.CurrentTextureSlot = 1;		

		if (s_RenderData.CircleBuffer.size() > 0)
		{
			s_RenderData.CircleShader->Bind();
			s_RenderData.BatchVertexBuffer->SetData(s_RenderData.CircleBuffer);
			glDrawElements(GL_TRIANGLES, (s_RenderData.CircleBuffer.size() / 4 * 6), GL_UNSIGNED_INT, nullptr);
			s_RenderData.CircleBuffer.clear();
		}


	
		if (s_RenderData.StencilRectBuffer.size() > 0)
		{
			//spdlog::info(__FUNCTION__);

			// initial draw
			s_RenderData.TextureShader->Bind();

			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LESS);
			glStencilFunc(GL_ALWAYS, 1, 0XFF);
			glStencilMask(0xFF);

			s_RenderData.BatchVertexBuffer->SetData(s_RenderData.StencilRectBuffer, 0, s_RenderData.StencilIndex);
			
			glDrawElements(GL_TRIANGLES, ((s_RenderData.StencilRectBuffer.size() / 4 * 6) * 0.5f), GL_UNSIGNED_INT, nullptr);


			// stencil draw
			glDisable(GL_DEPTH_TEST);
			glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
			glStencilMask(0x00);
			//
			s_RenderData.BatchVertexBuffer->SetData(s_RenderData.StencilRectBuffer, s_RenderData.StencilIndex, s_RenderData.StencilRectBuffer.size());
			glDrawElements(GL_TRIANGLES, ((s_RenderData.StencilRectBuffer.size() / 4 * 6) * 0.5f), GL_UNSIGNED_INT, nullptr);
			
			glStencilMask(0x00);
			glStencilFunc(GL_ALWAYS, 1, 0XFF);

			s_RenderData.StencilIndex = 0;
			s_RenderData.StencilRectBuffer.clear();

		}
		s_RenderData.QuadIndexCount = 0;

	}
	void BatchRenderer::DrawRectangle(const Rectangle& rectangle, const glm::vec2& quantity)
	{
		DrawRectangle(rectangle.GetPosition(), rectangle.GetScale(), rectangle.GetColor(), quantity);
	}
	void BatchRenderer::DrawRectangle(const glm::vec3& position, const glm::vec3& scale, const glm::vec4& color, const glm::vec2& quantity)
	{		
		constexpr float texIndex = 0.0f;
		glm::mat4 transform = glm::translate(glm::mat4(1.0f), position) * glm::scale(glm::mat4(1.0f), scale);
		
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[0], color, s_RenderData.NegativeTextureCoords, quantity, texIndex);
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[1], color, s_RenderData.NegativeTextureCoords, quantity, texIndex);
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[2], color, s_RenderData.NegativeTextureCoords, quantity, texIndex);
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[3], color, s_RenderData.NegativeTextureCoords, quantity, texIndex);
	}

	void BatchRenderer::StencilDraw(const glm::vec3& position, const glm::vec3& scale, const glm::vec4& color, const glm::vec4& outlineColor, const glm::vec2& quantity)
	{
		//s_RenderData.StencilVertexArray->Bind();
		//s_RenderData.StencilVertexBuffer->Bind();		
		//s_RenderData.TextureShader->Bind();

		constexpr float texIndex = 0.0f;

		glm::mat4 transform = glm::translate(glm::mat4(1.0f), position) * glm::scale(glm::mat4(1.0f), scale);

		s_RenderData.StencilRectBuffer.emplace(s_RenderData.StencilRectBuffer.begin()+s_RenderData.StencilIndex++,transform * s_RenderData.DefaultRectangleVertices[0], color, s_RenderData.NegativeTextureCoords, quantity, texIndex);
		s_RenderData.StencilRectBuffer.emplace(s_RenderData.StencilRectBuffer.begin()+s_RenderData.StencilIndex++,transform * s_RenderData.DefaultRectangleVertices[1], color, s_RenderData.NegativeTextureCoords, quantity, texIndex);
		s_RenderData.StencilRectBuffer.emplace(s_RenderData.StencilRectBuffer.begin()+s_RenderData.StencilIndex++,transform * s_RenderData.DefaultRectangleVertices[2], color, s_RenderData.NegativeTextureCoords, quantity, texIndex);
		s_RenderData.StencilRectBuffer.emplace(s_RenderData.StencilRectBuffer.begin()+s_RenderData.StencilIndex++,transform * s_RenderData.DefaultRectangleVertices[3], color, s_RenderData.NegativeTextureCoords, quantity, texIndex);

		glm::vec3 outlineScale = scale * 1.1f;
		transform = glm::translate(glm::mat4(1.0f), position) * glm::scale(glm::mat4(1.0f), outlineScale);
		
		s_RenderData.StencilRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[0], outlineColor, s_RenderData.NegativeTextureCoords, quantity);
		s_RenderData.StencilRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[1], outlineColor, s_RenderData.NegativeTextureCoords, quantity);
		s_RenderData.StencilRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[2], outlineColor, s_RenderData.NegativeTextureCoords, quantity);
		s_RenderData.StencilRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[3], outlineColor, s_RenderData.NegativeTextureCoords, quantity);

		
	}

	void BatchRenderer::StencilDraw(const glm::vec3& position, const glm::vec3& scale, const glm::vec4& color, const glm::vec4& outlineColor, std::string_view texture, const glm::vec2& quantity)
	{
		s_RenderData.TextureShader->Bind();

		auto tex = BatchTextureLoader::LoadTexture(texture);
		float texIndex = 0.0f;
		for (auto i = 1; i < s_RenderData.CurrentTextureSlot; i++)
		{
			if (s_RenderData.Textures[i]->GetTextureFile() == tex->GetTextureFile())
			{
				texIndex = static_cast<float>(i);
				break;
			}
		}
		if (texIndex == 0.0f)
		{
			texIndex = static_cast<float>(s_RenderData.CurrentTextureSlot);
			s_RenderData.Textures[s_RenderData.CurrentTextureSlot] = tex;
			s_RenderData.CurrentTextureSlot++;
		}

		glm::mat4 transform = glm::translate(glm::mat4(1.0f), position) * glm::scale(glm::mat4(1.0f), scale);

		s_RenderData.StencilRectBuffer.emplace(s_RenderData.StencilRectBuffer.begin() + s_RenderData.StencilIndex++, transform * s_RenderData.DefaultRectangleVertices[0], color, s_RenderData.DefaultTextureCoords[0], quantity, texIndex);
		s_RenderData.StencilRectBuffer.emplace(s_RenderData.StencilRectBuffer.begin() + s_RenderData.StencilIndex++, transform * s_RenderData.DefaultRectangleVertices[1], color, s_RenderData.DefaultTextureCoords[1], quantity, texIndex);
		s_RenderData.StencilRectBuffer.emplace(s_RenderData.StencilRectBuffer.begin() + s_RenderData.StencilIndex++, transform * s_RenderData.DefaultRectangleVertices[2], color, s_RenderData.DefaultTextureCoords[2], quantity, texIndex);
		s_RenderData.StencilRectBuffer.emplace(s_RenderData.StencilRectBuffer.begin() + s_RenderData.StencilIndex++, transform * s_RenderData.DefaultRectangleVertices[3], color, s_RenderData.DefaultTextureCoords[3], quantity, texIndex);

		glm::vec3 outlineScale = scale * 1.1f;
		transform = glm::translate(glm::mat4(1.0f), position) * glm::scale(glm::mat4(1.0f), outlineScale);
		
		s_RenderData.StencilRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[0], outlineColor, s_RenderData.NegativeTextureCoords, quantity, texIndex);
		s_RenderData.StencilRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[1], outlineColor, s_RenderData.NegativeTextureCoords, quantity, texIndex);
		s_RenderData.StencilRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[2], outlineColor, s_RenderData.NegativeTextureCoords, quantity, texIndex);
		s_RenderData.StencilRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[3], outlineColor, s_RenderData.NegativeTextureCoords, quantity, texIndex);
	}

	void BatchRenderer::StencilDraw(const Rectangle& rectangle, const glm::vec4& outlineColor, std::string_view texture, const glm::vec2& quantity)
	{
		StencilDraw(rectangle.GetPosition(), rectangle.GetScale(), rectangle.GetColor(),outlineColor, texture, quantity);
	}


	void BatchRenderer::DrawRectangle(const Rectangle& rectangle, std::string_view texture, const glm::vec2& quantity)
	{
		DrawRectangle(rectangle.GetPosition(), rectangle.GetScale(), rectangle.GetColor(), texture, quantity);
	}
	void BatchRenderer::DrawRectangle(const glm::vec3& position, const glm::vec3& scale, const glm::vec4& color, std::string_view texture, const glm::vec2& quantity)
	{	
		auto tex = BatchTextureLoader::LoadTexture(texture);
		float texIndex = 0.0f;
		for (auto i = 1; i < s_RenderData.CurrentTextureSlot; i++)
		{
			if (s_RenderData.Textures[i]->GetTextureFile() == tex->GetTextureFile())
			{
				texIndex = static_cast<float>(i);
				break;
			}
		}
		if (texIndex == 0.0f)
		{
			texIndex = static_cast<float>(s_RenderData.CurrentTextureSlot);
			s_RenderData.Textures[s_RenderData.CurrentTextureSlot] = tex;			
			s_RenderData.CurrentTextureSlot++;
		}				

		glm::mat4 transform = glm::translate(glm::mat4(1.0f), position) * glm::scale(glm::mat4(1.0f), scale);

		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[0], color, s_RenderData.DefaultTextureCoords[0], quantity, texIndex);
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[1], color, s_RenderData.DefaultTextureCoords[1], quantity, texIndex);
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[2], color, s_RenderData.DefaultTextureCoords[2], quantity, texIndex);
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[3], color, s_RenderData.DefaultTextureCoords[3], quantity, texIndex);
	}
	void BatchRenderer::DrawRotatedRectangle(const Rectangle& rectangle, const glm::vec2& quantity)
	{
		DrawRotatedRectangle(rectangle.GetPosition(), rectangle.GetScale(), rectangle.GetRotation(), rectangle.GetColor(), quantity);
	}
	void BatchRenderer::DrawRotatedRectangle(const glm::vec3& position, const glm::vec3& scale, float rotation, const glm::vec4& color, const glm::vec2& quantity)
	{
		constexpr float texIndex = 0.0f;

		glm::mat4 transform = glm::translate(glm::mat4(1.0f), position)
			* glm::rotate(glm::mat4(1.0f), glm::radians(rotation), glm::vec3(0.0f, 0.0f, 1.0f))
			* glm::scale(glm::mat4(1.0f), scale);

		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[0], color, s_RenderData.NegativeTextureCoords, quantity, texIndex);
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[1], color, s_RenderData.NegativeTextureCoords, quantity, texIndex);
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[2], color, s_RenderData.NegativeTextureCoords, quantity, texIndex);
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[3], color, s_RenderData.NegativeTextureCoords, quantity, texIndex);
	}
	void BatchRenderer::DrawRotatedRectangle(const Rectangle& rectangle, std::string_view texture, const glm::vec2& quantity)
	{
		DrawRotatedRectangle(rectangle.GetPosition(), rectangle.GetScale(), rectangle.GetRotation(), rectangle.GetColor(), texture, quantity);

	}
	void BatchRenderer::DrawRotatedRectangle(const glm::vec3& position, const glm::vec3& scale, float rotation, const glm::vec4& color, std::string_view texture, const glm::vec2& quantity)
	{
		auto tex = BatchTextureLoader::LoadTexture(texture);
		float texIndex = 0.0f;
		for (auto i = 1; i < s_RenderData.CurrentTextureSlot; i++)
		{
			if (s_RenderData.Textures[i]->GetTextureFile() == tex->GetTextureFile())
			{
				texIndex = static_cast<float>(i);
				break;
			}
		}
		if (texIndex == 0.0f)
		{
			texIndex = static_cast<float>(s_RenderData.CurrentTextureSlot);
			s_RenderData.Textures[s_RenderData.CurrentTextureSlot] = tex;			
			s_RenderData.CurrentTextureSlot++;
		}

		glm::mat4 transform = glm::translate(glm::mat4(1.0f), position)
			* glm::rotate(glm::mat4(1.0f), glm::radians(rotation), glm::vec3(0.0f, 0.0f, 1.0f))
			* glm::scale(glm::mat4(1.0f), scale);

		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[0], color, s_RenderData.DefaultTextureCoords[0], quantity, texIndex);
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[1], color, s_RenderData.DefaultTextureCoords[1], quantity, texIndex);
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[2], color, s_RenderData.DefaultTextureCoords[2], quantity, texIndex);
		s_RenderData.BatchRectBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[3], color, s_RenderData.DefaultTextureCoords[3], quantity, texIndex);
	}
	


	void BatchRenderer::DrawCircle(const Rectangle& rectangle, const glm::vec2& quantity)
	{
		DrawCircle(rectangle.GetPosition(), rectangle.GetScale(), rectangle.GetColor(), quantity);
	}

	// texCoords are used to store the position, quantity stores the scale
	// for some reason the fragment shader needs to use these values to update the cutoff and distance values
	void BatchRenderer::DrawCircle(const glm::vec3& position, const glm::vec3& scale, const glm::vec4& color, const glm::vec2& quantity)
	{
		if (s_RenderData.QuadIndexCount >= BatchRendererData::MaxIndices)
		{
			EndScene();
		}
		//s_RenderData.CircleShader->Bind();
		constexpr float texIndex = 0.0f;
		glm::vec2 scale1 = scale;
		glm::mat4 transform = glm::translate(glm::mat4(1.0f), position) * glm::scale(glm::mat4(1.0f), scale);

		s_RenderData.CircleBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[0], color, glm::vec2(position.x, position.y), glm::vec2(scale1.x, scale1.y), texIndex);
		s_RenderData.CircleBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[1], color, glm::vec2(position.x, position.y), glm::vec2(scale1.x, scale1.y), texIndex);
		s_RenderData.CircleBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[2], color, glm::vec2(position.x, position.y), glm::vec2(scale1.x, scale1.y), texIndex);
		s_RenderData.CircleBuffer.emplace_back(transform * s_RenderData.DefaultRectangleVertices[3], color, glm::vec2(position.x, position.y), glm::vec2(scale1.x, scale1.y), texIndex);
		//s_RenderData.TextureShader->Bind();
		s_RenderData.QuadIndexCount += 6;
	}

}