#include <cstdint>
#include <cassert>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

enum MdecBlockType
{
    MDEC_BLOCK_CR = 0,
    MDEC_BLOCK_CB = 1,
    MDEC_BLOCK_Y = 2
};

// Zigzag table
const uint8_t zagzig[64] = {
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63};

// Quantization tables for Y and Cr/Cb
const uint8_t y_quant_table[64] = {
    2, 16, 19, 22, 26, 27, 29, 34,
    16, 16, 22, 24, 27, 29, 34, 37,
    19, 22, 26, 27, 29, 34, 34, 38,
    22, 22, 26, 27, 29, 34, 37, 40,
    22, 26, 27, 29, 32, 35, 40, 48,
    26, 27, 29, 32, 35, 40, 48, 58,
    26, 27, 29, 34, 38, 46, 56, 69,
    27, 29, 35, 38, 46, 56, 69, 83};

const uint8_t c_quant_table[64] = {
    2, 16, 19, 22, 26, 27, 29, 34,
    16, 16, 22, 24, 27, 29, 34, 37,
    19, 22, 26, 27, 29, 34, 34, 38,
    22, 22, 26, 27, 29, 34, 37, 40,
    22, 26, 27, 29, 32, 35, 40, 48,
    26, 27, 29, 32, 35, 40, 48, 58,
    26, 27, 29, 34, 38, 46, 56, 69,
    27, 29, 35, 38, 46, 56, 69, 83};

const int16_t scale_table[64] = {
    23170, 23170, 23170, 23170, 23170, 23170, 23170, 23170, 32138, 27245, 18204, 6392, -6393,
    -18205, -27246, -32139, 30273, 12539, -12540, -30274, -30274, -12540, 12539, 30273, 27245,
    -6393, -32139, -18205, 18204, 32138, 6392, -27246, 23170, -23171, -23171, 23170, 23170,
    -23171, -23171, 23170, 18204, -32139, 6392, 27245, -27246, -6393, 32138, -18205, 12539,
    -30274, 30273, -12540, -12540, 30273, -30274, 12539, 6392, -18205, 27245, -32139, 32138,
    -27246, 18204, -6393};

// Perform IDCT on 8x8 block
void idct_core(int16_t src[8][8], int16_t dst[8][8])
{
    for (int i = 0; i < 2; i++)
    {
        for (int x = 0; x < 8; x++)
        {
            for (int y = 0; y < 8; y++)
            {
                int32_t sum = 0;
                for (int z = 0; z < 8; z++)
                    sum += (int32_t)src[z][y] * (int32_t)(scale_table[x + z * 8] >> 3);
                dst[y][x] = (sum + 0x0fff) >> 13;
            }
        }
        if (i == 0)
            for (int j = 0; j < 8; j++)
                std::swap(src[j], dst[j]);
    }
}

int16_t quantize_dc(uint16_t val, uint8_t quant)
{
    int16_t _val = (int16_t)(val << 6) >> 6;
    int32_t c;
    if (quant == 0)
        c = (int32_t)_val << 1;
    else
        c = (int32_t)_val * (int32_t)quant;
    return (int16_t)std::min(std::max(c, -0x4000), 0x3fff);
}

int16_t quantize_ac(uint16_t val, uint8_t quant, uint8_t qScale)
{
    int16_t _val = (int16_t)(val << 6) >> 6;
    int32_t c;
    if ((int32_t)quant * (int32_t)qScale == 0)
        c = (int32_t)_val << 1;
    else
        c = ((int32_t)_val * (int32_t)quant * (int32_t)qScale + 4) >> 3;
    return (int16_t)std::min(std::max(c, -0x4000), 0x3fff);
}

// Decode RLE data to block
void rle_decode(uint16_t **data, int16_t *blk, MdecBlockType block_type)
{
    // Select quantization table based on block type
    const uint8_t *qt = (block_type == MDEC_BLOCK_Y) ? y_quant_table : c_quant_table;
    int32_t c = 0;

    // Initialize block to zeros
    for (int i = 0; i < 64; i++)
        blk[i] = 0;

    // Look for start of block (skip FE00 markers)
    uint16_t n = *(*data)++;
    int k = 0;
    while (n == 0xfe00)
        n = *(*data)++;

    // Extract q_scale and DC value
    uint8_t q_scale = (n >> 10) & 0x3f;
    uint16_t val = n & 0x3ff;

    // Store DC value
    blk[zagzig[k]] = quantize_dc(val, qt[k]);

    // Process AC coefficients
    k++;
    n = *(*data)++;

    while (k < 64)
    {
        // Get run length
        int run = (n >> 10) & 0x3f;
        k += run;

        if (k >= 64)
            break;

        // Get AC value
        val = n & 0x3ff;

        // Apply quantization and scaling
        blk[zagzig[k]] = quantize_ac(val, qt[k], q_scale);

        k++;
        if (k >= 64)
            break;

        // Get next code
        n = *(*data)++;

        // Check for end of block
        if (n == 0xfe00)
            break;
    }
}

// Process a single 8x8 block
void process_mdec_block(uint16_t **rle_data, int16_t output[8][8], MdecBlockType block_type)
{
    int16_t rle_decoded[64] = {0};
    int16_t dct_block[8][8] = {0};

    // Decode RLE data
    rle_decode(rle_data, rle_decoded, block_type);

    // Convert 1D array to 8x8 block
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            dct_block[i][j] = rle_decoded[i * 8 + j];

    // Apply IDCT
    idct_core(dct_block, output);
}

int8_t sign_extend_9bits_clamp_8bits(int32_t val)
{
    int16_t signed_val = static_cast<int16_t>(static_cast<uint16_t>(val) << 7) >> 7;
    return (int8_t)std::min(std::max(signed_val, (int16_t)-128), (int16_t)127);
}

// Convert YUV to RGB
void yuv_to_rgb(int16_t yBlk[8][8], int16_t cbBlk[8][8], int16_t crBlk[8][8],
                uint8_t xx, uint8_t yy, uint8_t xOff, uint8_t yOff,
                uint8_t *dst, int stride)
{
    for (int y = 0; y < 8; y++)
    {
        for (int x = 0; x < 8; x++)
        {
            int32_t Y = yBlk[y][x];

            // Calculate chroma indices based on offsets and ensure they're within 4x4 region
            int cb_x = (x + xOff) / 2;
            int cb_y = (y + yOff) / 2;
            int32_t Cb = cbBlk[cb_y][cb_x]; // Chroma at half resolution with proper offset
            int32_t Cr = crBlk[cb_y][cb_x];

            // Calculate RGB values
            int32_t R = (int32_t)(Y + (1.402 * Cr));
            int32_t G = (int32_t)(Y + (-0.3437 * Cb) + (-0.7143 * Cr));
            int32_t B = (int32_t)(Y + (1.772 * Cb));

            // Store RGB values in output buffer
            int offset = (y + yy + yOff) * stride * 3 + (x + xx + xOff) * 3;
            dst[offset] = sign_extend_9bits_clamp_8bits(R) ^ 0x80;
            dst[offset + 1] = sign_extend_9bits_clamp_8bits(G) ^ 0x80;
            dst[offset + 2] = sign_extend_9bits_clamp_8bits(B) ^ 0x80;
        }
    }
}

// Process a 16x16 macroblock
void process_macroblock(uint16_t **rle_data, uint8_t *output_image,
                        int image_width, int mb_x, int mb_y)
{
    int16_t y_blocks[4][8][8];
    int16_t cb_block[8][8];
    int16_t cr_block[8][8];

    // Process Cr block (chrominance red)
    process_mdec_block(rle_data, cr_block, MDEC_BLOCK_CR);

    // Process Cb block (chrominance blue)
    process_mdec_block(rle_data, cb_block, MDEC_BLOCK_CB);

    // Process Y blocks (luminance)
    for (int i = 0; i < 4; i++)
        process_mdec_block(rle_data, y_blocks[i], MDEC_BLOCK_Y);

    // Convert YUV to RGB for each 8x8 block within the macroblock
    yuv_to_rgb(y_blocks[0], cb_block, cr_block, mb_x, mb_y, 0, 0, output_image, image_width);
    yuv_to_rgb(y_blocks[1], cb_block, cr_block, mb_x, mb_y, 8, 0, output_image, image_width); // Order differs from PSX-SPX ???
    yuv_to_rgb(y_blocks[2], cb_block, cr_block, mb_x, mb_y, 0, 8, output_image, image_width);
    yuv_to_rgb(y_blocks[3], cb_block, cr_block, mb_x, mb_y, 8, 8, output_image, image_width);
}

// Main MDEC decoder function
void decode_mdec_image(uint16_t **data, int data_size, int width, int height, const char *output_file)
{
    // Allocate memory for output image (RGB format)
    uint8_t *output_image = new uint8_t[width * height * 3];

    // Process macroblocks in column-major order
    for (int x = 0; x < width; x += 16)
        for (int y = 0; y < height; y += 16)
            process_macroblock(data, output_image, width, x, y);

    // Save decoded image
    if (stbi_write_png(output_file, width, height, 3, output_image, width * 3))
        std::cout << "Successfully saved PNG image!" << std::endl;
    else
        std::cerr << "Failed to save PNG image!" << std::endl;
    delete[] output_image;
}

// Simple command-line interface
int main(int argc, char *argv[])
{
    // Parse command line arguments
    const char *input_file = argv[1]; // "../../../../test.bin";
    int width = std::stoi(argv[2]);   // 256;
    int height = std::stoi(argv[3]);  // 192;
    const char *output_file = "output.png";

    // Read input file
    std::ifstream file(input_file, std::ios::binary | std::ios::ate);
    if (!file)
    {
        std::cerr << "Error: Could not open input file " << input_file << std::endl;
        return 1;
    }

    // Get file size and allocate buffer
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint16_t> buffer(size / sizeof(uint16_t));
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size))
    {
        std::cerr << "Error: Could not read input file" << std::endl;
        return 1;
    }

    // Decode the image
    uint16_t *buf_ptr = buffer.data();
    decode_mdec_image(&buf_ptr, (int)buffer.size(), width, height, output_file);

    return 0;
}
