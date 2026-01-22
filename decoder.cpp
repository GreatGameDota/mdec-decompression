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

const uint8_t zigzag[64] = {
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63};

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

const double scalefactor[8] = {1.000000000, 1.387039845, 1.306562965, 1.175875602, 1.000000000, 0.785694958, 0.541196100, 0.275899379};
const double scalezag[64] = {
    0.125, 0.17338, 0.17338, 0.16332, 0.240485, 0.16332, 0.146984, 0.226532,
    0.226532, 0.146984, 0.125, 0.203873, 0.213388, 0.203873, 0.125, 0.0982119,
    0.17338, 0.192044, 0.192044, 0.17338, 0.0982119, 0.0676495, 0.136224, 0.16332,
    0.172835, 0.16332, 0.136224, 0.0676495, 0.0344874, 0.0938326, 0.12832, 0.146984,
    0.146984, 0.12832, 0.0938326, 0.0344874, 0.0478354, 0.0883883, 0.115485, 0.125,
    0.115485, 0.0883883, 0.0478354, 0.04506, 0.0795474, 0.0982119, 0.0982119, 0.0795474,
    0.04506, 0.0405529, 0.0676495, 0.0771646, 0.0676495, 0.0405529, 0.0344874, 0.0531519,
    0.0531519, 0.0344874, 0.0270966, 0.0366117, 0.0270966, 0.0186645, 0.0186645, 0.00951506};

bool earlyTerminate = false;

// Perform IDCT on 8x8 block
void idct_core(int16_t src[8][8], int16_t dst[8][8])
{
    for (int pass = 0; pass < 2; pass++)
    {
        for (int i = 0; i < 8; i++)
        {
            // Quick fill if AC coefficients are zero
            if (src[1][i] == 0 && src[2][i] == 0 && src[3][i] == 0 &&
                src[4][i] == 0 && src[5][i] == 0 && src[6][i] == 0 && src[7][i] == 0)
            {
                for (int j = 0; j < 8; j++)
                    dst[i][j] = src[0][i];
            }
            else
            {
                double z10, z11, z12, z13, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

                z10 = (double)src[0][i] + src[4][i];
                z11 = (double)src[0][i] - src[4][i];
                z13 = (double)src[2][i] + src[6][i];
                z12 = (double)src[2][i] - src[6][i];

                z12 = (1.414213562 * z12) - z13;

                tmp0 = z10 + z13;
                tmp3 = z10 - z13;
                tmp1 = z11 + z12;
                tmp2 = z11 - z12;

                z13 = (double)src[3][i] + src[5][i];
                z10 = (double)src[3][i] - src[5][i];
                z11 = (double)src[1][i] + src[7][i];
                z12 = (double)src[1][i] - src[7][i];

                double z5 = 1.847759065 * (z12 - z10);

                tmp7 = z11 + z13;
                tmp6 = (2.613125930 * z10) + z5 - tmp7;
                tmp5 = (1.414213562 * (z11 - z13)) - tmp6;
                tmp4 = (1.082392200 * z12) - z5 + tmp5;

                dst[i][0] = (int16_t)(tmp0 + tmp7);
                dst[i][7] = (int16_t)(tmp0 - tmp7);
                dst[i][1] = (int16_t)(tmp1 + tmp6);
                dst[i][6] = (int16_t)(tmp1 - tmp6);
                dst[i][2] = (int16_t)(tmp2 + tmp5);
                dst[i][5] = (int16_t)(tmp2 - tmp5);
                dst[i][4] = (int16_t)(tmp3 + tmp4);
                dst[i][3] = (int16_t)(tmp3 - tmp4);
            }
        }

        if (pass == 0)
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
void rle_decode(uint16_t **data, int16_t *blk, MdecBlockType block_type, uint16_t *end)
{
    // Select quantization table based on block type
    const uint8_t *qt = (block_type == MDEC_BLOCK_Y) ? y_quant_table : c_quant_table;
    int32_t c = 0;

    // Initialize block to zeros
    for (int i = 0; i < 64; i++)
        blk[i] = 0;

    if (*data >= end)
    {
        earlyTerminate = true;
        return;
    }

    // Look for start of block (skip FE00 markers)
    uint16_t n = *(*data)++;
    int k = 0;
    while (n == 0xfe00 && *data < end)
        n = *(*data)++;

    if (*data >= end)
    {
        earlyTerminate = true;
        return;
    }

    // Extract q_scale and DC value
    uint8_t q_scale = (n >> 10) & 0x3f;
    uint16_t val = n & 0x3ff;

    // Store DC value
    blk[zagzig[k]] = (int16_t)((double)quantize_dc(val, qt[k]) * scalezag[k]);

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
        blk[zagzig[k]] = (int16_t)((double)quantize_ac(val, qt[k], q_scale) * scalezag[k]);

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
void process_mdec_block(uint16_t **rle_data, int16_t output[8][8], MdecBlockType block_type,
                        uint16_t *end)
{
    int16_t rle_decoded[64] = {0};
    int16_t dct_block[8][8] = {0};

    // Decode RLE data
    rle_decode(rle_data, rle_decoded, block_type, end);

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
void process_macroblock(uint16_t **rle_data, uint8_t *output_image, uint16_t *end,
                        int image_width, int mb_x, int mb_y)
{
    int16_t y_blocks[4][8][8];
    int16_t cb_block[8][8];
    int16_t cr_block[8][8];

    // Process Cr block (chrominance red)
    process_mdec_block(rle_data, cr_block, MDEC_BLOCK_CR, end);

    // Process Cb block (chrominance blue)
    process_mdec_block(rle_data, cb_block, MDEC_BLOCK_CB, end);

    // Process Y blocks (luminance)
    for (int i = 0; i < 4; i++)
        process_mdec_block(rle_data, y_blocks[i], MDEC_BLOCK_Y, end);

    // Convert YUV to RGB for each 8x8 block within the macroblock
    yuv_to_rgb(y_blocks[0], cb_block, cr_block, mb_x, mb_y, 0, 0, output_image, image_width);
    yuv_to_rgb(y_blocks[1], cb_block, cr_block, mb_x, mb_y, 8, 0, output_image, image_width); // Order differs from PSX-SPX ???
    yuv_to_rgb(y_blocks[2], cb_block, cr_block, mb_x, mb_y, 0, 8, output_image, image_width);
    yuv_to_rgb(y_blocks[3], cb_block, cr_block, mb_x, mb_y, 8, 8, output_image, image_width);
}

// Main MDEC decoder function
void decode_mdec_image(uint16_t **data, uint16_t *end, int width, int height, const char *output_file)
{
    // Allocate memory for output image (RGB format)
    uint8_t *output_image = new uint8_t[width * height * 3];

    // Process macroblocks in column-major order
    std::vector<uint8_t *> patches;
    while (*data < end) // Decode image
    {
        uint8_t *patch = new uint8_t[16 * 16 * 3];
        earlyTerminate = false;
        process_macroblock(data, patch, end, 16, 0, 0);
        if (!earlyTerminate)
            patches.push_back(patch);
        else
            delete[] patch;
    }
    printf("Decoded %zu patches\n", patches.size());
    // Reconstruct full image from patches
    int patches_per_column = (height + 15) / 16; // Ensure proper handling of non-multiples of 16
    for (int i = 0; i < (int)patches.size(); i++)
    {
        int patch_x = (i / patches_per_column) * 16;
        int patch_y = (i % patches_per_column) * 16;

        for (int y = 0; y < 16; y++)
        {
            if (patch_y + y >= height)
                break;
            for (int x = 0; x < 16; x++)
            {
                if (patch_x + x >= width)
                    break;
                int dst_offset = ((patch_y + y) * width + (patch_x + x)) * 3;
                int src_offset = (y * 16 + x) * 3;
                output_image[dst_offset] = patches[i][src_offset];
                output_image[dst_offset + 1] = patches[i][src_offset + 1];
                output_image[dst_offset + 2] = patches[i][src_offset + 2];
            }
        }
        delete[] patches[i];
    }
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
    decode_mdec_image(&buf_ptr, buf_ptr + buffer.size(), width, height, output_file);

    return 0;
}
