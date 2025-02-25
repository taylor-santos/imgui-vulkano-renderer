mod cache;
mod shader;

use bytemuck::{Pod, Zeroable};
use cache::DescriptorSetCache;
use imgui::internal::RawWrapper;
use imgui::{DrawCmd, DrawCmdParams, DrawVert, TextureId, Textures};
use std::{fmt, sync::Arc};
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::SubpassBeginInfo;
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::ImageType::Dim2d;
use vulkano::image::ImageUsage;
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, ColorBlendAttachmentState};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexDefinition;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::graphics::{vertex_input, GraphicsPipelineCreateInfo};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::Subpass;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::InputAssemblyState,
            viewport::{Scissor, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    sync::GpuFuture,
};

#[derive(Default, Debug, Copy, Clone, vertex_input::Vertex, Zeroable, Pod)]
#[repr(C)]
struct Vertex {
    #[format(R32G32_SFLOAT)]
    pub pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    pub col: [u8; 4],
}

impl From<DrawVert> for Vertex {
    fn from(v: DrawVert) -> Vertex {
        Vertex {
            pos: v.pos,
            uv: v.uv,
            col: v.col,
        }
    }
}

#[derive(Debug)]
pub enum RendererError {
    BadTexture(TextureId),
    BadImageDimensions([u32; 3]),
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadTexture(t) => {
                write!(f, "The Texture ID could not be found: {:?}", t)
            }
            Self::BadImageDimensions(d) => {
                write!(f, "Image Dimensions not supported (must be Dim2d): {:?}", d)
            }
        }
    }
}

impl std::error::Error for RendererError {}

pub type Texture = (Arc<ImageView>, Arc<Sampler>);

pub struct Allocators {
    pub descriptor_sets: Arc<StandardDescriptorSetAllocator>,
    pub memory: Arc<StandardMemoryAllocator>,
    pub command_buffers: Arc<StandardCommandBufferAllocator>,
}

pub struct Renderer {
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    font_texture: Texture,
    textures: Textures<Texture>,

    allocators: Allocators,

    descriptor_set_cache: DescriptorSetCache,
}

impl Renderer {
    /// Initialize the renderer object, including vertex buffers, ImGui font textures,
    /// and the Vulkan graphics pipeline.
    ///
    /// ---
    ///
    /// `ctx`: the ImGui `Context` object
    ///
    /// `device`: the Vulkano `Device` object for the device you want to render the UI on.
    ///
    /// `queue`: the Vulkano `Queue` object for the queue the font atlas texture will be created on.
    ///
    /// `format`: the Vulkano `Format` that the render pass will use when storing the frame in the target image.
    pub fn init(
        ctx: &mut imgui::Context,
        device: Arc<Device>,
        queue: Arc<Queue>,
        format: Format,
        gamma: Option<f32>,
        allocators: Option<Allocators>,
    ) -> Result<Renderer, Box<dyn std::error::Error>> {
        let allocators = allocators.unwrap_or_else(|| Allocators {
            descriptor_sets: Arc::new(StandardDescriptorSetAllocator::new(
                device.clone(),
                Default::default(),
            )),
            memory: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
            command_buffers: Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                StandardCommandBufferAllocatorCreateInfo::default(),
            )),
        });

        let vs = shader::vs::load(device.clone())?
            .entry_point("main")
            .ok_or("Failed to load vertex shader")?;
        let fs = shader::fs::load(device.clone())?
            .specialize([(0, gamma.unwrap_or(1.0).into())].into_iter().collect())?
            .entry_point("main")
            .ok_or("Failed to load fragment shader")?;

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: format,
                    samples: 1,
                    load_op: Load,
                    store_op: Store,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )?;

        let pipeline = {
            let subpass =
                Subpass::from(render_pass.clone(), 0).ok_or("Failed to create subpass")?;
            let vertex_input_state =
                <Vertex as vertex_input::Vertex>::per_vertex().definition(&vs)?;
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ]
            .into_iter()
            .collect();
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())?,
            )?;
            let color_blend_state = ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::alpha()),
                    ..Default::default()
                },
            );
            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages,
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        scissors: vec![Scissor::default()].into(),
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    color_blend_state: Some(color_blend_state),
                    dynamic_state: [
                        vulkano::pipeline::DynamicState::Viewport,
                        vulkano::pipeline::DynamicState::Scissor,
                    ]
                    .into_iter()
                    .collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )?
        };

        let textures = Textures::new();
        let font_texture = Self::upload_font_texture(
            ctx.fonts(),
            device.clone(),
            queue.clone(),
            &allocators,
        )?;

        ctx.set_renderer_name(Some(format!(
            "imgui-vulkano-renderer {}",
            env!("CARGO_PKG_VERSION")
        )));

        Ok(Renderer {
            render_pass,
            pipeline,
            font_texture,
            textures,
            allocators,
            descriptor_set_cache: DescriptorSetCache::default(),
        })
    }

    /// Appends the draw commands for the UI frame to an `AutoCommandBufferBuilder`.
    ///
    /// ---
    ///
    /// `cmd_buf_builder`: An `AutoCommandBufferBuilder` from vulkano to add commands to
    ///
    /// `device`: the Vulkano `Device` object for the device you want to render the UI on
    ///
    /// `queue`: the Vulkano `Queue` object for buffer creation
    ///
    /// `target`: the target image to render to
    ///
    /// `draw_data`: the ImGui `DrawData` that each UI frame creates
    pub fn draw_commands(
        &mut self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        target: Arc<ImageView>,
        draw_data: &imgui::DrawData,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
        if !(fb_width > 0.0 && fb_height > 0.0) {
            return Ok(());
        }
        let left = draw_data.display_pos[0];
        let right = draw_data.display_pos[0] + draw_data.display_size[0];
        let top = draw_data.display_pos[1];
        let bottom = draw_data.display_pos[1] + draw_data.display_size[1];

        let pc = shader::vs::VertPC {
            matrix: [
                [(2.0 / (right - left)), 0.0, 0.0, 0.0],
                [0.0, (2.0 / (bottom - top)), 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [
                    (right + left) / (left - right),
                    (top + bottom) / (top - bottom),
                    0.0,
                    1.0,
                ],
            ],
        };

        let dims = target.image().extent();

        let clip_off = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;

        let layout = &self.pipeline.layout().set_layouts()[0];

        // Creating a new Framebuffer every frame isn't ideal, but according to this thread,
        // it also isn't really an issue on desktop GPUs:
        // https://github.com/GameTechDev/IntroductionToVulkan/issues/20
        // This might be a good target for optimizations in the future though.
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![target],
                ..Default::default()
            },
        )?;

        let info = RenderPassBeginInfo {
            clear_values: vec![None],
            ..RenderPassBeginInfo::framebuffer(framebuffer)
        };

        cmd_buf_builder
            .begin_render_pass(info, SubpassBeginInfo::default())?
            .bind_pipeline_graphics(self.pipeline.clone())?;

        if draw_data.draw_lists_count() > 0 { // Until https://github.com/imgui-rs/imgui-rs/pull/779 is published to crates.io
            for draw_list in draw_data.draw_lists() {
                let vertices: Vec<Vertex> = draw_list
                    .vtx_buffer()
                    .iter()
                    .map(|v| (*v).into())
                    .collect();
                let indices = draw_list.idx_buffer().to_vec();

                let vertex_buffer = Buffer::from_iter(
                    self.allocators.memory.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    vertices,
                )?;

                let index_buffer = Buffer::from_iter(
                    self.allocators.memory.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::INDEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    indices,
                )?;

                for cmd in draw_list.commands() {
                    match cmd {
                        DrawCmd::Elements {
                            count,
                            cmd_params:
                                DrawCmdParams {
                                    clip_rect,
                                    texture_id,
                                    idx_offset,
                                    // vtx_offset,
                                    ..
                                },
                        } => {
                            let clip_rect = [
                                (clip_rect[0] - clip_off[0]) * clip_scale[0],
                                (clip_rect[1] - clip_off[1]) * clip_scale[1],
                                (clip_rect[2] - clip_off[0]) * clip_scale[0],
                                (clip_rect[3] - clip_off[1]) * clip_scale[1],
                            ];

                            if clip_rect[0] < fb_width
                                && clip_rect[1] < fb_height
                                && clip_rect[2] >= 0.0
                                && clip_rect[3] >= 0.0
                            {
                                let set = self.descriptor_set_cache.get_or_insert(
                                    texture_id,
                                    |texture_id| {
                                        let (img, sampler) = Self::lookup_texture(
                                            &self.textures,
                                            &self.font_texture,
                                            texture_id,
                                        )?
                                        .clone();

                                        DescriptorSet::new(
                                            self.allocators.descriptor_sets.clone(),
                                            layout.clone(),
                                            [WriteDescriptorSet::image_view_sampler(0, img, sampler)],
                                            [],
                                        )
                                        .map_err(Into::into)
                                    },
                                )?;

                                cmd_buf_builder
                                    .bind_descriptor_sets(
                                        PipelineBindPoint::Graphics,
                                        self.pipeline.layout().clone(),
                                        0,
                                        set.clone(),
                                    )?
                                    .set_scissor(
                                        0,
                                        std::iter::once(Scissor {
                                            offset: [
                                                f32::max(0.0, clip_rect[0]).floor() as u32,
                                                f32::max(0.0, clip_rect[1]).floor() as u32,
                                            ],
                                            extent: [
                                                (clip_rect[2] - clip_rect[0]).abs().ceil() as u32,
                                                (clip_rect[3] - clip_rect[1]).abs().ceil() as u32,
                                            ],
                                        })
                                        .collect(),
                                    )?
                                    .set_viewport(
                                        0,
                                        std::iter::once(Viewport {
                                            offset: [0.0, 0.0],
                                            extent: [dims[0] as f32, dims[1] as f32],
                                            depth_range: 0.0..=1.0,
                                        })
                                        .collect(),
                                    )?
                                    .bind_vertex_buffers(0, vertex_buffer.clone())?
                                    .bind_index_buffer(index_buffer.clone())?
                                    .push_constants(self.pipeline.layout().clone(), 0, pc)?;
                                unsafe {
                                    cmd_buf_builder.draw_indexed(
                                        count as u32,
                                        1,
                                        idx_offset as u32,
                                        0,
                                        0,
                                    )
                                }?;
                            }
                        }
                        DrawCmd::ResetRenderState => (),
                        DrawCmd::RawCallback { callback, raw_cmd } => unsafe {
                            callback(draw_list.raw(), raw_cmd)
                        },
                    }
                }
            }
        }
        cmd_buf_builder.end_render_pass(Default::default())?;

        Ok(())
    }

    /// Update the ImGui font atlas texture.
    ///
    /// ---
    ///
    /// `ctx`: the ImGui `Context` object
    ///
    /// `device`: the Vulkano `Device` object for the device you want to render the UI on.
    ///
    /// `queue`: the Vulkano `Queue` object for the queue the font atlas texture will be created on.
    pub fn reload_font_texture(
        &mut self,
        ctx: &mut imgui::Context,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.descriptor_set_cache.clear_font_texture();
        self.font_texture =
            Self::upload_font_texture(ctx.fonts(), device, queue, &self.allocators)?;
        Ok(())
    }

    /// Get the texture library that the renderer uses
    pub fn textures_mut(&mut self) -> &mut Textures<Texture> {
        // make sure to recreate descriptors if necessary
        self.descriptor_set_cache.clear();
        &mut self.textures
    }

    /// Get the texture library that the renderer uses
    pub fn textures(&self) -> &Textures<Texture> {
        &self.textures
    }

    fn upload_font_texture(
        fonts: &mut imgui::FontAtlas,
        device: Arc<Device>,
        queue: Arc<Queue>,
        allocators: &Allocators,
    ) -> Result<Texture, Box<dyn std::error::Error>> {
        let texture = fonts.build_rgba32_texture();

        let mut builder = AutoCommandBufferBuilder::primary(
            allocators.command_buffers.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        let image = Image::new(
            allocators.memory.clone(),
            ImageCreateInfo {
                image_type: Dim2d,
                extent: [texture.width, texture.height, 1],
                array_layers: 1,
                mip_levels: 1,
                format: Format::R8G8B8A8_SRGB,
                usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )?;

        let upload_buffer: Subbuffer<[u8]> = Buffer::from_iter(
            allocators.memory.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            texture.data.iter().copied(),
        )?;

        builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            upload_buffer,
            image.clone(),
        ))?;

        let command_buffer = builder.build()?;

        command_buffer
            .execute(queue)?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        let sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear())?;

        fonts.tex_id = TextureId::from(usize::MAX);
        Ok((ImageView::new_default(image)?, sampler))
    }

    fn lookup_texture<'a>(
        textures: &'a Textures<Texture>,
        font_texture: &'a Texture,
        texture_id: TextureId,
    ) -> Result<&'a Texture, RendererError> {
        if texture_id.id() == usize::MAX {
            Ok(font_texture)
        } else if let Some(texture) = textures.get(texture_id) {
            Ok(texture)
        } else {
            Err(RendererError::BadTexture(texture_id))
        }
    }
}
