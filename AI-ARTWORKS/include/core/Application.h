/**
 * @file Application.h
 * @brief Main application class for AI-Artworks
 * @author AI-Artworks Team
 * @version 1.0.0
 */

#pragma once

#include <memory>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>

#ifdef _WIN32
    #include <windows.h>
#endif

// Forward declarations to avoid circular dependencies
class AIProcessor;
class GraphicsEngine;
class AudioProcessor;
class AssetManager;

/**
 * @brief Main application singleton class
 * 
 * This class manages the entire application lifecycle, coordinates
 * between different subsystems, and provides a central point of control.
 */
class Application {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the Application instance
     */
    static Application& getInstance();

    /**
     * @brief Initialize the application
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Run the main application loop
     * @return Exit code (0 for success)
     */
    int run();

    /**
     * @brief Shutdown the application
     */
    void shutdown();

    /**
     * @brief Check if the application is running
     * @return true if running, false otherwise
     */
    bool isRunning() const { return m_isRunning.load(); }

    /**
     * @brief Request application shutdown
     */
    void requestShutdown() { m_shouldShutdown.store(true); }

    // Subsystem accessors
    AIProcessor& getAIProcessor();
    GraphicsEngine& getGraphicsEngine();
    AudioProcessor& getAudioProcessor();
    AssetManager& getAssetManager();

    /**
     * @brief Get application version
     * @return Version string
     */
    static std::string getVersion() { return "1.0.0"; }

    /**
     * @brief Get build information
     * @return Build info string
     */
    static std::string getBuildInfo();

private:
    // Singleton pattern - private constructor/destructor
    Application();
    ~Application();
    
    // Prevent copying
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    // Initialization methods
    bool initializeSubsystems();
    bool initializeGraphics();
    bool initializeAudio();
    bool initializeAI();
    bool initializeAssets();

    // Main loop methods
    void processEvents();
    void update(float deltaTime);
    void render();

    // Error handling
    void handleError(const std::string& error);
    void logSystemInfo();

    // Platform-specific methods
#ifdef _WIN32
    bool initializeWindows();
    void handleWindowsMessage(UINT message, WPARAM wParam, LPARAM lParam);
#endif

private:
    // Core state
    std::atomic<bool> m_isInitialized{false};
    std::atomic<bool> m_isRunning{false};
    std::atomic<bool> m_shouldShutdown{false};
    
    // Thread safety
    mutable std::mutex m_mutex;
    
    // Subsystems (using unique_ptr for PIMPL and proper cleanup)
    std::unique_ptr<AIProcessor> m_aiProcessor;
    std::unique_ptr<GraphicsEngine> m_graphicsEngine;
    std::unique_ptr<AudioProcessor> m_audioProcessor;
    std::unique_ptr<AssetManager> m_assetManager;
    
    // Timing
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    float m_deltaTime{0.0f};
    
    // Performance metrics
    uint64_t m_frameCount{0};
    float m_averageFPS{0.0f};
    
#ifdef _WIN32
    HWND m_windowHandle{nullptr};
    HDC m_deviceContext{nullptr};
#endif
};