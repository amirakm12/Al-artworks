/**
 * @file Application.cpp
 * @brief Implementation of the main application class
 * @author AI-Artworks Team
 * @version 1.0.0
 */

#include "core/Application.h"
#include "ai/AIProcessor.h"
#include "graphics/GraphicsEngine.h"
#include "audio/AudioProcessor.h"
#include "utils/AssetManager.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <sstream>

#ifdef HAS_FMT
    #include <fmt/format.h>
    #define LOG_INFO(msg, ...) fmt::print(fg(fmt::color::cyan), "[APP] " msg "\n", ##__VA_ARGS__)
    #define LOG_ERROR(msg, ...) fmt::print(fg(fmt::color::red), "[APP ERROR] " msg "\n", ##__VA_ARGS__)
    #define LOG_SUCCESS(msg, ...) fmt::print(fg(fmt::color::green), "[APP] " msg "\n", ##__VA_ARGS__)
    #define LOG_WARNING(msg, ...) fmt::print(fg(fmt::color::yellow), "[APP WARNING] " msg "\n", ##__VA_ARGS__)
#else
    #define LOG_INFO(msg, ...) std::cout << "[APP] " << msg << std::endl
    #define LOG_ERROR(msg, ...) std::cerr << "[APP ERROR] " << msg << std::endl
    #define LOG_SUCCESS(msg, ...) std::cout << "[APP] " << msg << std::endl
    #define LOG_WARNING(msg, ...) std::cout << "[APP WARNING] " << msg << std::endl
#endif

#ifdef HAS_SPDLOG
    #include <spdlog/spdlog.h>
#endif

Application& Application::getInstance() {
    static Application instance;
    return instance;
}

Application::Application() {
    LOG_INFO("Application instance created");
}

Application::~Application() {
    if (m_isRunning.load()) {
        shutdown();
    }
    LOG_INFO("Application instance destroyed");
}

bool Application::initialize() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_isInitialized.load()) {
        LOG_WARNING("Application already initialized");
        return true;
    }
    
    try {
        LOG_INFO("Initializing AI-Artworks Application...");
        
        // Log system information
        logSystemInfo();
        
        // Initialize platform-specific components
#ifdef _WIN32
        if (!initializeWindows()) {
            handleError("Failed to initialize Windows-specific components");
            return false;
        }
#endif
        
        // Initialize subsystems
        if (!initializeSubsystems()) {
            handleError("Failed to initialize subsystems");
            return false;
        }
        
        // Set timing baseline
        m_lastFrameTime = std::chrono::high_resolution_clock::now();
        
        m_isInitialized.store(true);
        LOG_SUCCESS("Application initialized successfully");
        return true;
    }
    catch (const std::exception& e) {
        handleError("Exception during initialization: " + std::string(e.what()));
        return false;
    }
    catch (...) {
        handleError("Unknown exception during initialization");
        return false;
    }
}

bool Application::initializeSubsystems() {
    LOG_INFO("Initializing subsystems...");
    
    try {
        // Initialize Asset Manager first (other systems may depend on it)
        if (!initializeAssets()) {
            return false;
        }
        
        // Initialize AI Processor
        if (!initializeAI()) {
            LOG_WARNING("AI Processor initialization failed - continuing without AI features");
        }
        
        // Initialize Graphics Engine
        if (!initializeGraphics()) {
            LOG_WARNING("Graphics Engine initialization failed - continuing with limited graphics");
        }
        
        // Initialize Audio Processor
        if (!initializeAudio()) {
            LOG_WARNING("Audio Processor initialization failed - continuing without audio");
        }
        
        LOG_SUCCESS("Subsystems initialized");
        return true;
    }
    catch (const std::exception& e) {
        handleError("Exception initializing subsystems: " + std::string(e.what()));
        return false;
    }
}

bool Application::initializeAssets() {
    try {
        LOG_INFO("Initializing Asset Manager...");
        m_assetManager = std::make_unique<AssetManager>();
        
        if (!m_assetManager->initialize()) {
            LOG_ERROR("Failed to initialize Asset Manager");
            return false;
        }
        
        LOG_SUCCESS("Asset Manager initialized");
        return true;
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception initializing Asset Manager: {}", e.what());
        return false;
    }
}

bool Application::initializeAI() {
    try {
        LOG_INFO("Initializing AI Processor...");
        m_aiProcessor = std::make_unique<AIProcessor>();
        
        if (!m_aiProcessor->initialize()) {
            LOG_ERROR("Failed to initialize AI Processor");
            return false;
        }
        
        LOG_SUCCESS("AI Processor initialized");
        return true;
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception initializing AI Processor: {}", e.what());
        return false;
    }
}

bool Application::initializeGraphics() {
    try {
        LOG_INFO("Initializing Graphics Engine...");
        m_graphicsEngine = std::make_unique<GraphicsEngine>();
        
        if (!m_graphicsEngine->initialize()) {
            LOG_ERROR("Failed to initialize Graphics Engine");
            return false;
        }
        
        LOG_SUCCESS("Graphics Engine initialized");
        return true;
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception initializing Graphics Engine: {}", e.what());
        return false;
    }
}

bool Application::initializeAudio() {
    try {
        LOG_INFO("Initializing Audio Processor...");
        m_audioProcessor = std::make_unique<AudioProcessor>();
        
        if (!m_audioProcessor->initialize()) {
            LOG_ERROR("Failed to initialize Audio Processor");
            return false;
        }
        
        LOG_SUCCESS("Audio Processor initialized");
        return true;
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception initializing Audio Processor: {}", e.what());
        return false;
    }
}

#ifdef _WIN32
bool Application::initializeWindows() {
    try {
        LOG_INFO("Initializing Windows-specific components...");
        
        // Set high DPI awareness
        SetProcessDPIAware();
        
        // Set process priority for better performance
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        
        // Enable Windows 10 dark mode if available
        HKEY hKey;
        DWORD dwValue = 1;
        if (RegOpenKeyEx(HKEY_CURRENT_USER, 
                        L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize", 
                        0, KEY_SET_VALUE, &hKey) == ERROR_SUCCESS) {
            RegSetValueEx(hKey, L"AppsUseLightTheme", 0, REG_DWORD, 
                         reinterpret_cast<const BYTE*>(&dwValue), sizeof(dwValue));
            RegCloseKey(hKey);
        }
        
        LOG_SUCCESS("Windows-specific components initialized");
        return true;
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception initializing Windows components: {}", e.what());
        return false;
    }
}
#endif

int Application::run() {
    if (!m_isInitialized.load()) {
        LOG_ERROR("Application not initialized - cannot run");
        return 1;
    }
    
    LOG_INFO("Starting main application loop...");
    m_isRunning.store(true);
    
    try {
        // Main application loop
        while (m_isRunning.load() && !m_shouldShutdown.load()) {
            // Calculate delta time
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                currentTime - m_lastFrameTime);
            m_deltaTime = duration.count() / 1000000.0f; // Convert to seconds
            m_lastFrameTime = currentTime;
            
            // Process events
            processEvents();
            
            // Update application state
            update(m_deltaTime);
            
            // Render frame
            render();
            
            // Update performance metrics
            m_frameCount++;
            if (m_frameCount % 60 == 0) { // Update FPS every 60 frames
                m_averageFPS = 1.0f / m_deltaTime;
                
#ifdef HAS_SPDLOG
                spdlog::debug("FPS: {:.1f}, Frame: {}", m_averageFPS, m_frameCount);
#endif
            }
            
            // Prevent 100% CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        LOG_INFO("Main loop ended gracefully");
        return 0;
    }
    catch (const std::exception& e) {
        handleError("Exception in main loop: " + std::string(e.what()));
        return 2;
    }
    catch (...) {
        handleError("Unknown exception in main loop");
        return 3;
    }
}

void Application::processEvents() {
    try {
#ifdef _WIN32
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                requestShutdown();
                break;
            }
            
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            
            handleWindowsMessage(msg.message, msg.wParam, msg.lParam);
        }
#endif
        
        // Process other events here (input, network, etc.)
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception processing events: {}", e.what());
    }
}

void Application::update(float deltaTime) {
    try {
        // Update subsystems
        if (m_aiProcessor) {
            m_aiProcessor->update(deltaTime);
        }
        
        if (m_graphicsEngine) {
            m_graphicsEngine->update(deltaTime);
        }
        
        if (m_audioProcessor) {
            m_audioProcessor->update(deltaTime);
        }
        
        if (m_assetManager) {
            m_assetManager->update(deltaTime);
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception during update: {}", e.what());
    }
}

void Application::render() {
    try {
        if (m_graphicsEngine) {
            m_graphicsEngine->beginFrame();
            m_graphicsEngine->render();
            m_graphicsEngine->endFrame();
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception during render: {}", e.what());
    }
}

void Application::shutdown() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_isInitialized.load()) {
        return;
    }
    
    LOG_INFO("Shutting down application...");
    m_isRunning.store(false);
    
    try {
        // Shutdown subsystems in reverse order
        if (m_audioProcessor) {
            m_audioProcessor->shutdown();
            m_audioProcessor.reset();
        }
        
        if (m_graphicsEngine) {
            m_graphicsEngine->shutdown();
            m_graphicsEngine.reset();
        }
        
        if (m_aiProcessor) {
            m_aiProcessor->shutdown();
            m_aiProcessor.reset();
        }
        
        if (m_assetManager) {
            m_assetManager->shutdown();
            m_assetManager.reset();
        }
        
        m_isInitialized.store(false);
        LOG_SUCCESS("Application shutdown complete");
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception during shutdown: {}", e.what());
    }
}

#ifdef _WIN32
void Application::handleWindowsMessage(UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_CLOSE:
        case WM_DESTROY:
            requestShutdown();
            break;
            
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                requestShutdown();
            }
            break;
            
        default:
            break;
    }
}
#endif

// Subsystem accessors with error handling
AIProcessor& Application::getAIProcessor() {
    if (!m_aiProcessor) {
        throw std::runtime_error("AI Processor not initialized");
    }
    return *m_aiProcessor;
}

GraphicsEngine& Application::getGraphicsEngine() {
    if (!m_graphicsEngine) {
        throw std::runtime_error("Graphics Engine not initialized");
    }
    return *m_graphicsEngine;
}

AudioProcessor& Application::getAudioProcessor() {
    if (!m_audioProcessor) {
        throw std::runtime_error("Audio Processor not initialized");
    }
    return *m_audioProcessor;
}

AssetManager& Application::getAssetManager() {
    if (!m_assetManager) {
        throw std::runtime_error("Asset Manager not initialized");
    }
    return *m_assetManager;
}

std::string Application::getBuildInfo() {
    std::ostringstream oss;
    oss << "AI-Artworks v" << getVersion() << "\n";
    oss << "Built: " << __DATE__ << " " << __TIME__ << "\n";
    oss << "Compiler: ";
    
#ifdef _MSC_VER
    oss << "MSVC " << _MSC_VER;
#elif defined(__GNUC__)
    oss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__;
#elif defined(__clang__)
    oss << "Clang " << __clang_major__ << "." << __clang_minor__;
#else
    oss << "Unknown";
#endif
    
    oss << "\nPlatform: ";
#ifdef _WIN32
    oss << "Windows";
#ifdef _WIN64
    oss << " x64";
#else
    oss << " x86";
#endif
#else
    oss << "Unknown";
#endif
    
    return oss.str();
}

void Application::logSystemInfo() {
    LOG_INFO("System Information:");
    LOG_INFO(getBuildInfo());
    
#ifdef _WIN32
    // Get Windows version
    OSVERSIONINFOEX osvi;
    ZeroMemory(&osvi, sizeof(OSVERSIONINFOEX));
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
    
    if (GetVersionEx((OSVERSIONINFO*)&osvi)) {
        LOG_INFO("Windows Version: {}.{}.{}", 
                osvi.dwMajorVersion, osvi.dwMinorVersion, osvi.dwBuildNumber);
    }
    
    // Get memory info
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        double totalGB = static_cast<double>(memInfo.ullTotalPhys) / (1024 * 1024 * 1024);
        double availGB = static_cast<double>(memInfo.ullAvailPhys) / (1024 * 1024 * 1024);
        LOG_INFO("Memory: {:.1f} GB total, {:.1f} GB available", totalGB, availGB);
    }
#endif
    
    // CPU info
    unsigned int numCores = std::thread::hardware_concurrency();
    LOG_INFO("CPU Cores: {}", numCores);
}

void Application::handleError(const std::string& error) {
    LOG_ERROR(error);
    
#ifdef HAS_SPDLOG
    spdlog::error(error);
#endif
    
    // Could add additional error handling here:
    // - Write to crash log
    // - Send telemetry
    // - Show error dialog
}