/**
 * @file AIProcessor.cpp
 * @brief Implementation of AI processing system
 * @author AI-Artworks Team
 * @version 1.0.0
 */

#include "ai/AIProcessor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <filesystem>

#ifdef _WIN32
    #include <windows.h>
    #include <d3d11.h>
    #include <dxgi.h>
    #pragma comment(lib, "d3d11.lib")
    #pragma comment(lib, "dxgi.lib")
#endif

#ifdef HAS_FMT
    #include <fmt/format.h>
    #define LOG_INFO(msg, ...) fmt::print(fg(fmt::color::cyan), "[AI] " msg "\n", ##__VA_ARGS__)
    #define LOG_ERROR(msg, ...) fmt::print(fg(fmt::color::red), "[AI ERROR] " msg "\n", ##__VA_ARGS__)
    #define LOG_SUCCESS(msg, ...) fmt::print(fg(fmt::color::green), "[AI] " msg "\n", ##__VA_ARGS__)
    #define LOG_WARNING(msg, ...) fmt::print(fg(fmt::color::yellow), "[AI WARNING] " msg "\n", ##__VA_ARGS__)
#else
    #define LOG_INFO(msg, ...) std::cout << "[AI] " << msg << std::endl
    #define LOG_ERROR(msg, ...) std::cerr << "[AI ERROR] " << msg << std::endl
    #define LOG_SUCCESS(msg, ...) std::cout << "[AI] " << msg << std::endl
    #define LOG_WARNING(msg, ...) std::cout << "[AI WARNING] " << msg << std::endl
#endif

#ifdef HAS_SPDLOG
    #include <spdlog/spdlog.h>
#endif

// PIMPL implementation structure
struct AIProcessor::Impl {
    std::string lastError;
    std::string modelPath;
    std::string modelInfo;
    
    // Model parameters
    float temperature = 0.7f;
    float topP = 0.9f;
    int topK = 40;
    
    // Mock model state (replace with actual llama.cpp integration)
    bool mockModelLoaded = false;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point lastProcessingStart;
    
    // GPU information
    std::string gpuInfo;
    size_t gpuMemoryMB = 0;
    bool gpuAvailable = false;
};

AIProcessor::AIProcessor() : m_impl(std::make_unique<Impl>()) {
    LOG_INFO("AI Processor created");
}

AIProcessor::~AIProcessor() {
    if (m_initialized.load()) {
        shutdown();
    }
    LOG_INFO("AI Processor destroyed");
}

bool AIProcessor::initialize() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_initialized.load()) {
        LOG_WARNING("AI Processor already initialized");
        return true;
    }
    
    try {
        LOG_INFO("Initializing AI Processor...");
        
        // Check system requirements
        unsigned int numCores = std::thread::hardware_concurrency();
        LOG_INFO("Detected {} CPU cores", numCores);
        
        // Initialize GPU support if available
#ifdef _WIN32
        if (!initializeWindowsGPU()) {
            LOG_WARNING("GPU initialization failed - using CPU only");
        }
#endif
        
        // Check for GPU availability
        if (isGPUAvailable()) {
            size_t gpuMem = getAvailableGPUMemory();
            LOG_SUCCESS("GPU acceleration available - {} MB VRAM", gpuMem);
            m_impl->gpuAvailable = true;
            m_impl->gpuMemoryMB = gpuMem;
        } else {
            LOG_INFO("GPU acceleration not available - using CPU only");
            m_impl->gpuAvailable = false;
        }
        
        // Initialize default configuration
        m_currentConfig.threads = (numCores > 4) ? numCores - 2 : numCores; // Leave some cores for system
        m_currentConfig.contextSize = 4096;
        m_currentConfig.temperature = 0.7f;
        m_currentConfig.maxTokens = 512;
        m_currentConfig.useGPU = m_impl->gpuAvailable;
        
        m_initialized.store(true);
        LOG_SUCCESS("AI Processor initialized successfully");
        return true;
    }
    catch (const std::exception& e) {
        handleError("Exception during AI Processor initialization: " + std::string(e.what()));
        return false;
    }
    catch (...) {
        handleError("Unknown exception during AI Processor initialization");
        return false;
    }
}

void AIProcessor::shutdown() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_initialized.load()) {
        return;
    }
    
    LOG_INFO("Shutting down AI Processor...");
    
    try {
        // Unload model if loaded
        if (m_modelLoaded.load()) {
            unloadModel();
        }
        
        // Wait for any processing to complete
        if (m_processingThread.joinable()) {
            m_processingThread.join();
        }
        
        m_initialized.store(false);
        LOG_SUCCESS("AI Processor shutdown complete");
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception during AI Processor shutdown: {}", e.what());
    }
}

void AIProcessor::update(float deltaTime) {
    // Update any background processing, statistics, etc.
    // This is called each frame from the main application loop
    
    if (!m_initialized.load()) {
        return;
    }
    
    // Update statistics, cleanup, etc.
    // For now, this is mostly a placeholder
}

bool AIProcessor::loadModel(const std::string& modelPath, const AIModelConfig& config) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_initialized.load()) {
        handleError("AI Processor not initialized");
        return false;
    }
    
    if (m_processing.load()) {
        handleError("Cannot load model while processing");
        return false;
    }
    
    try {
        LOG_INFO("Loading AI model: {}", modelPath);
        
        // Validate model file
        if (!validateModelFile(modelPath)) {
            handleError("Model file validation failed: " + modelPath);
            return false;
        }
        
        // Unload existing model if any
        if (m_modelLoaded.load()) {
            unloadModel();
        }
        
        // Store configuration
        m_currentConfig = config;
        m_currentConfig.modelPath = modelPath;
        
        // Apply configuration defaults if not set
        if (m_currentConfig.threads <= 0) {
            m_currentConfig.threads = std::thread::hardware_concurrency();
        }
        
        LOG_INFO("Model configuration:");
        LOG_INFO("  Context size: {}", m_currentConfig.contextSize);
        LOG_INFO("  Threads: {}", m_currentConfig.threads);
        LOG_INFO("  Temperature: {:.2f}", m_currentConfig.temperature);
        LOG_INFO("  Max tokens: {}", m_currentConfig.maxTokens);
        LOG_INFO("  Use GPU: {}", m_currentConfig.useGPU ? "Yes" : "No");
        
        // TODO: Replace with actual llama.cpp model loading
        // For now, this is a mock implementation
        auto start = std::chrono::high_resolution_clock::now();
        
        // Simulate model loading time
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Mock successful loading
        m_impl->mockModelLoaded = true;
        m_impl->modelPath = modelPath;
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Extract model info from filename for now
        std::filesystem::path path(modelPath);
        m_impl->modelInfo = "Model: " + path.filename().string() + 
                           "\nSize: " + std::to_string(std::filesystem::file_size(path) / (1024*1024)) + " MB" +
                           "\nLoading time: " + std::to_string(duration.count()) + " ms";
        
        m_modelLoaded.store(true);
        logModelInfo();
        
        LOG_SUCCESS("Model loaded successfully in {} ms", duration.count());
        return true;
    }
    catch (const std::exception& e) {
        handleError("Exception loading model: " + std::string(e.what()));
        return false;
    }
    catch (...) {
        handleError("Unknown exception loading model");
        return false;
    }
}

void AIProcessor::unloadModel() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_modelLoaded.load()) {
        return;
    }
    
    LOG_INFO("Unloading AI model...");
    
    try {
        // Wait for any processing to complete
        while (m_processing.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // TODO: Replace with actual llama.cpp model unloading
        m_impl->mockModelLoaded = false;
        m_impl->modelPath.clear();
        m_impl->modelInfo.clear();
        
        m_modelLoaded.store(false);
        LOG_SUCCESS("Model unloaded successfully");
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception unloading model: {}", e.what());
    }
}

AIProcessingResult AIProcessor::processText(const std::string& prompt, int maxTokens) {
    AIProcessingResult result;
    
    if (!m_modelLoaded.load()) {
        result.error = "No model loaded";
        return result;
    }
    
    if (prompt.empty()) {
        result.error = "Empty prompt";
        return result;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    m_processing.store(true);
    
    try {
        LOG_INFO("Processing text prompt: '{}'", prompt.substr(0, 50) + (prompt.length() > 50 ? "..." : ""));
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // TODO: Replace with actual llama.cpp inference
        // For now, this is a mock implementation
        
        int tokensToGenerate = (maxTokens > 0) ? maxTokens : m_currentConfig.maxTokens;
        
        // Simulate processing time based on token count
        int processingTimeMs = tokensToGenerate * 10; // 10ms per token simulation
        std::this_thread::sleep_for(std::chrono::milliseconds(processingTimeMs));
        
        // Generate mock response based on prompt content
        std::string response;
        if (prompt.find("artwork") != std::string::npos || prompt.find("art") != std::string::npos) {
            response = "A stunning digital artwork featuring vibrant colors and abstract geometric patterns. "
                      "The composition flows with dynamic energy, blending modern techniques with classical "
                      "artistic principles. The piece evokes a sense of movement and emotion through its "
                      "carefully balanced use of light and shadow, creating depth and visual interest.";
        } else if (prompt.find("description") != std::string::npos) {
            response = "This is a detailed description generated by the AI model. The content is rich "
                      "and informative, providing comprehensive insights into the requested topic. "
                      "The response demonstrates the model's ability to understand context and generate "
                      "relevant, coherent text that addresses the user's query effectively.";
        } else {
            response = "This is a response generated by the AI model based on your prompt. The model "
                      "has processed your input and generated this relevant text output. The response "
                      "demonstrates the AI's natural language processing capabilities.";
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        result.success = true;
        result.result = response;
        result.processingTime = duration.count() / 1000.0f;
        result.tokensGenerated = tokensToGenerate;
        
        // Update statistics
        updateStatistics(result);
        
        LOG_SUCCESS("Text processing completed in {:.2f}s, {} tokens generated", 
                   result.processingTime, result.tokensGenerated);
    }
    catch (const std::exception& e) {
        result.error = "Exception during text processing: " + std::string(e.what());
        LOG_ERROR(result.error);
    }
    catch (...) {
        result.error = "Unknown exception during text processing";
        LOG_ERROR(result.error);
    }
    
    m_processing.store(false);
    return result;
}

AIProcessingResult AIProcessor::processTextStreaming(const std::string& prompt, 
                                                   AIStreamCallback callback,
                                                   int maxTokens) {
    AIProcessingResult result;
    
    if (!callback) {
        result.error = "No callback provided for streaming";
        return result;
    }
    
    // For now, simulate streaming by calling the regular processText and breaking it into chunks
    result = processText(prompt, maxTokens);
    
    if (result.success && callback) {
        // Simulate streaming by sending the result in chunks
        std::string text = result.result;
        size_t chunkSize = 10; // Characters per chunk
        
        for (size_t i = 0; i < text.length(); i += chunkSize) {
            std::string chunk = text.substr(i, chunkSize);
            bool isComplete = (i + chunkSize >= text.length());
            
            callback(chunk, isComplete);
            
            // Simulate streaming delay
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    return result;
}

AIProcessingResult AIProcessor::generateArtworkDescription(const std::string& artworkType,
                                                         const std::string& style,
                                                         const std::vector<std::string>& additionalPrompts) {
    std::string prompt = buildArtworkPrompt(artworkType, style, additionalPrompts);
    return processText(prompt);
}

AIProcessingResult AIProcessor::generateArtworkMetadata(const std::string& description) {
    std::string prompt = "Generate JSON metadata for this artwork description: " + description + 
                        "\nInclude fields: title, artist, medium, dimensions, year, description, tags, style, color_palette";
    
    AIProcessingResult result = processText(prompt);
    
    if (result.success) {
        // Ensure the result is valid JSON (mock for now)
        result.result = R"({
    "title": "AI Generated Artwork",
    "artist": "AI-Artworks System",
    "medium": "Digital",
    "dimensions": "1920x1080",
    "year": "2024",
    "description": ")" + description + R"(",
    "tags": ["ai-generated", "digital-art", "abstract"],
    "style": "contemporary",
    "color_palette": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"]
})";
    }
    
    return result;
}

std::string AIProcessor::getModelInfo() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_modelLoaded.load()) {
        return "No model loaded";
    }
    
    return m_impl->modelInfo;
}

std::string AIProcessor::getProcessingStats() const {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    
    std::ostringstream oss;
    oss << "Processing Statistics:\n";
    oss << "  Total requests: " << m_totalProcessingRequests << "\n";
    oss << "  Total tokens: " << m_totalTokensGenerated << "\n";
    oss << "  Total time: " << m_totalProcessingTime << "s\n";
    oss << "  Average tokens/sec: " << m_averageTokensPerSecond << "\n";
    
    return oss.str();
}

void AIProcessor::setModelParameters(float temperature, float topP, int topK) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    m_impl->temperature = std::clamp(temperature, 0.0f, 2.0f);
    m_impl->topP = std::clamp(topP, 0.0f, 1.0f);
    m_impl->topK = std::max(1, topK);
    
    LOG_INFO("Model parameters updated - Temperature: {:.2f}, TopP: {:.2f}, TopK: {}", 
             m_impl->temperature, m_impl->topP, m_impl->topK);
}

bool AIProcessor::isGPUAvailable() {
    // TODO: Implement actual GPU detection
    // For now, return a mock result based on platform
#ifdef _WIN32
    return true; // Assume Windows has some form of GPU
#else
    return false;
#endif
}

size_t AIProcessor::getAvailableGPUMemory() {
    // TODO: Implement actual GPU memory detection
    // For now, return a mock value
#ifdef _WIN32
    return 8192; // Mock 8GB VRAM
#else
    return 0;
#endif
}

// Private methods implementation

bool AIProcessor::validateModelFile(const std::string& modelPath) {
    try {
        if (!std::filesystem::exists(modelPath)) {
            LOG_ERROR("Model file does not exist: {}", modelPath);
            return false;
        }
        
        auto fileSize = std::filesystem::file_size(modelPath);
        if (fileSize < 1024 * 1024) { // Less than 1MB
            LOG_ERROR("Model file too small: {} bytes", fileSize);
            return false;
        }
        
        // Check file extension
        std::filesystem::path path(modelPath);
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        if (extension != ".gguf" && extension != ".bin") {
            LOG_WARNING("Unexpected model file extension: {}", extension);
        }
        
        LOG_INFO("Model file validation passed - Size: {} MB", fileSize / (1024*1024));
        return true;
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception validating model file: {}", e.what());
        return false;
    }
}

std::string AIProcessor::buildArtworkPrompt(const std::string& artworkType,
                                          const std::string& style,
                                          const std::vector<std::string>& additionalPrompts) {
    std::ostringstream oss;
    oss << "Generate a detailed description for a " << artworkType << " artwork";
    
    if (!style.empty()) {
        oss << " in " << style << " style";
    }
    
    oss << ". The description should be vivid and artistic, suitable for an art gallery catalog.";
    
    if (!additionalPrompts.empty()) {
        oss << " Additional requirements: ";
        for (size_t i = 0; i < additionalPrompts.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << additionalPrompts[i];
        }
        oss << ".";
    }
    
    return oss.str();
}

void AIProcessor::updateStatistics(const AIProcessingResult& result) {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    
    m_totalProcessingRequests++;
    m_totalTokensGenerated += result.tokensGenerated;
    m_totalProcessingTime += result.processingTime;
    
    if (m_totalProcessingTime > 0) {
        m_averageTokensPerSecond = static_cast<float>(m_totalTokensGenerated) / m_totalProcessingTime;
    }
}

void AIProcessor::logModelInfo() {
    LOG_INFO("Model Information:");
    LOG_INFO(m_impl->modelInfo);
}

void AIProcessor::handleError(const std::string& error) {
    m_impl->lastError = error;
    LOG_ERROR(error);
    
#ifdef HAS_SPDLOG
    spdlog::error("[AI] {}", error);
#endif
}

std::string AIProcessor::getLastError() const {
    return m_impl->lastError;
}

#ifdef _WIN32
bool AIProcessor::initializeWindowsGPU() {
    try {
        // Initialize DirectX for GPU detection
        IDXGIFactory* pFactory = nullptr;
        HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory);
        
        if (SUCCEEDED(hr)) {
            IDXGIAdapter* pAdapter = nullptr;
            UINT adapterIndex = 0;
            
            while (pFactory->EnumAdapters(adapterIndex, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
                DXGI_ADAPTER_DESC adapterDesc;
                pAdapter->GetDesc(&adapterDesc);
                
                // Convert wide string to regular string
                char adapterName[256];
                WideCharToMultiByte(CP_UTF8, 0, adapterDesc.Description, -1, 
                                   adapterName, sizeof(adapterName), NULL, NULL);
                
                LOG_INFO("GPU {}: {} - {} MB VRAM", 
                        adapterIndex, adapterName, 
                        adapterDesc.DedicatedVideoMemory / (1024*1024));
                
                m_impl->gpuInfo = adapterName;
                m_impl->gpuMemoryMB = adapterDesc.DedicatedVideoMemory / (1024*1024);
                
                pAdapter->Release();
                adapterIndex++;
            }
            
            pFactory->Release();
            return adapterIndex > 0;
        }
        
        return false;
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception initializing Windows GPU: {}", e.what());
        return false;
    }
}
#endif