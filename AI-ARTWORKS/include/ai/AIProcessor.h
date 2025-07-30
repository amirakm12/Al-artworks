/**
 * @file AIProcessor.h
 * @brief AI processing system for artwork generation and text processing
 * @author AI-Artworks Team
 * @version 1.0.0
 */

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>
#include <thread>

/**
 * @brief AI model configuration structure
 */
struct AIModelConfig {
    std::string modelPath;
    int contextSize = 4096;
    int threads = -1; // -1 = auto-detect
    float temperature = 0.7f;
    int maxTokens = 512;
    bool useGPU = true;
    std::string systemPrompt;
};

/**
 * @brief AI processing result structure
 */
struct AIProcessingResult {
    bool success = false;
    std::string result;
    std::string error;
    float processingTime = 0.0f;
    int tokensGenerated = 0;
};

/**
 * @brief Callback function type for streaming responses
 */
using AIStreamCallback = std::function<void(const std::string& token, bool isComplete)>;

/**
 * @brief Main AI processing class
 * 
 * This class handles loading and running AI models (GGUF format),
 * text generation, and artwork description generation.
 */
class AIProcessor {
public:
    AIProcessor();
    ~AIProcessor();

    /**
     * @brief Initialize the AI processor
     * @return true if successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Shutdown the AI processor
     */
    void shutdown();

    /**
     * @brief Update the AI processor (called each frame)
     * @param deltaTime Time since last update in seconds
     */
    void update(float deltaTime);

    /**
     * @brief Load an AI model from file
     * @param modelPath Path to the GGUF model file
     * @param config Optional configuration parameters
     * @return true if loaded successfully, false otherwise
     */
    bool loadModel(const std::string& modelPath, const AIModelConfig& config = {});

    /**
     * @brief Unload the current model
     */
    void unloadModel();

    /**
     * @brief Check if a model is currently loaded
     * @return true if model is loaded, false otherwise
     */
    bool isModelLoaded() const { return m_modelLoaded.load(); }

    /**
     * @brief Process text with the loaded model
     * @param prompt Input text prompt
     * @param maxTokens Maximum tokens to generate (0 = use model default)
     * @return Processing result with generated text
     */
    AIProcessingResult processText(const std::string& prompt, int maxTokens = 0);

    /**
     * @brief Process text with streaming callback
     * @param prompt Input text prompt
     * @param callback Function called for each generated token
     * @param maxTokens Maximum tokens to generate (0 = use model default)
     * @return Processing result
     */
    AIProcessingResult processTextStreaming(const std::string& prompt, 
                                          AIStreamCallback callback,
                                          int maxTokens = 0);

    /**
     * @brief Generate artwork description
     * @param artworkType Type of artwork (e.g., "abstract", "landscape", "portrait")
     * @param style Art style (e.g., "impressionist", "digital", "watercolor")
     * @param additionalPrompts Additional descriptive prompts
     * @return Generated artwork description
     */
    AIProcessingResult generateArtworkDescription(const std::string& artworkType,
                                                const std::string& style = "",
                                                const std::vector<std::string>& additionalPrompts = {});

    /**
     * @brief Generate artwork metadata
     * @param description Artwork description
     * @return JSON-formatted metadata
     */
    AIProcessingResult generateArtworkMetadata(const std::string& description);

    /**
     * @brief Get model information
     * @return String containing model details
     */
    std::string getModelInfo() const;

    /**
     * @brief Get processing statistics
     * @return String containing performance stats
     */
    std::string getProcessingStats() const;

    /**
     * @brief Set model parameters
     * @param temperature Randomness (0.0-2.0)
     * @param topP Nucleus sampling parameter (0.0-1.0)
     * @param topK Top-k sampling parameter
     */
    void setModelParameters(float temperature, float topP = 0.9f, int topK = 40);

    /**
     * @brief Check if GPU acceleration is available
     * @return true if GPU can be used, false otherwise
     */
    static bool isGPUAvailable();

    /**
     * @brief Get available GPU memory in MB
     * @return GPU memory in MB, 0 if no GPU
     */
    static size_t getAvailableGPUMemory();

private:
    // Internal implementation details (PIMPL pattern)
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    // State management
    std::atomic<bool> m_initialized{false};
    std::atomic<bool> m_modelLoaded{false};
    std::atomic<bool> m_processing{false};
    
    // Thread safety
    mutable std::mutex m_mutex;
    std::thread m_processingThread;
    
    // Configuration
    AIModelConfig m_currentConfig;
    
    // Statistics
    mutable std::mutex m_statsMutex;
    uint64_t m_totalProcessingRequests{0};
    uint64_t m_totalTokensGenerated{0};
    float m_totalProcessingTime{0.0f};
    float m_averageTokensPerSecond{0.0f};

    // Internal methods
    bool validateModelFile(const std::string& modelPath);
    std::string buildArtworkPrompt(const std::string& artworkType,
                                 const std::string& style,
                                 const std::vector<std::string>& additionalPrompts);
    void updateStatistics(const AIProcessingResult& result);
    void logModelInfo();
    
    // Error handling
    void handleError(const std::string& error);
    std::string getLastError() const;
    
    // Platform-specific initialization
#ifdef _WIN32
    bool initializeWindowsGPU();
#endif
};