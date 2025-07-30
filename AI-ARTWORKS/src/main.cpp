/**
 * @file main.cpp
 * @brief Main entry point for AI-Artworks application
 * @author AI-Artworks Team
 * @version 1.0.0
 */

#include <iostream>
#include <memory>
#include <exception>
#include <string>
#include <chrono>

#ifdef _WIN32
    #include <windows.h>
    #include <io.h>
    #include <fcntl.h>
    #include <conio.h>
#endif

// Conditional includes based on available libraries
#ifdef HAS_FMT
    #include <fmt/format.h>
    #include <fmt/color.h>
    #define LOG_INFO(msg, ...) fmt::print(fg(fmt::color::cyan), "[INFO] " msg "\n", ##__VA_ARGS__)
    #define LOG_ERROR(msg, ...) fmt::print(fg(fmt::color::red), "[ERROR] " msg "\n", ##__VA_ARGS__)
    #define LOG_SUCCESS(msg, ...) fmt::print(fg(fmt::color::green), "[SUCCESS] " msg "\n", ##__VA_ARGS__)
    #define LOG_WARNING(msg, ...) fmt::print(fg(fmt::color::yellow), "[WARNING] " msg "\n", ##__VA_ARGS__)
#else
    #define LOG_INFO(msg, ...) std::cout << "[INFO] " << msg << std::endl
    #define LOG_ERROR(msg, ...) std::cerr << "[ERROR] " << msg << std::endl
    #define LOG_SUCCESS(msg, ...) std::cout << "[SUCCESS] " << msg << std::endl
    #define LOG_WARNING(msg, ...) std::cout << "[WARNING] " << msg << std::endl
#endif

#ifdef HAS_SPDLOG
    #include <spdlog/spdlog.h>
    #include <spdlog/sinks/stdout_color_sinks.h>
    #include <spdlog/sinks/rotating_file_sink.h>
#endif

#include "core/Application.h"

/**
 * @brief Configure console for Windows
 */
void ConfigureWindowsConsole() {
#ifdef _WIN32
    try {
        // Enable UTF-8 output
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
        
        // Enable ANSI color codes
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut != INVALID_HANDLE_VALUE) {
            DWORD dwMode = 0;
            if (GetConsoleMode(hOut, &dwMode)) {
                dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                SetConsoleMode(hOut, dwMode);
            }
        }
        
        // Set console title
        SetConsoleTitleA("AI-Artworks - AI-Powered Artwork Generation");
        
        LOG_SUCCESS("Windows console configured successfully");
    }
    catch (const std::exception& e) {
        LOG_WARNING("Failed to configure Windows console: {}", e.what());
    }
#endif
}

/**
 * @brief Initialize logging system
 */
void InitializeLogging() {
    try {
#ifdef HAS_SPDLOG
        // Create console sink with colors
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
        
        // Create rotating file sink
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            "logs/ai-artworks.log", 1024 * 1024 * 5, 3);
        file_sink->set_level(spdlog::level::debug);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");
        
        // Create logger with both sinks
        std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
        auto logger = std::make_shared<spdlog::logger>("ai-artworks", sinks.begin(), sinks.end());
        logger->set_level(spdlog::level::debug);
        
        // Set as default logger
        spdlog::set_default_logger(logger);
        spdlog::flush_every(std::chrono::seconds(3));
        
        LOG_SUCCESS("Advanced logging system initialized");
#else
        LOG_INFO("Using basic logging (spdlog not available)");
#endif
    }
    catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize logging: {}", e.what());
    }
}

/**
 * @brief Display application banner
 */
void DisplayBanner() {
    const std::string banner = R"(
╔═══════════════════════════════════════════════════════════════╗
║                          AI-ARTWORKS                          ║
║                 AI-Powered Artwork Generation                 ║
║                        Version 1.0.0                         ║
╚═══════════════════════════════════════════════════════════════╝
)";
    
#ifdef HAS_FMT
    fmt::print(fg(fmt::color::magenta), "{}", banner);
#else
    std::cout << banner << std::endl;
#endif
}

/**
 * @brief Check system requirements
 */
bool CheckSystemRequirements() {
    LOG_INFO("Checking system requirements...");
    
    bool allRequirementsMet = true;
    
#ifdef _WIN32
    // Check Windows version
    OSVERSIONINFOEX osvi;
    ZeroMemory(&osvi, sizeof(OSVERSIONINFOEX));
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
    
    if (GetVersionEx((OSVERSIONINFO*)&osvi)) {
        if (osvi.dwMajorVersion >= 10) {
            LOG_SUCCESS("Windows version: {}.{} (Compatible)", osvi.dwMajorVersion, osvi.dwMinorVersion);
        } else {
            LOG_WARNING("Windows version: {}.{} (Windows 10+ recommended)", osvi.dwMajorVersion, osvi.dwMinorVersion);
        }
    }
    
    // Check available memory
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        DWORDLONG totalPhysMem = memInfo.ullTotalPhys;
        double totalGB = static_cast<double>(totalPhysMem) / (1024 * 1024 * 1024);
        
        if (totalGB >= 8.0) {
            LOG_SUCCESS("System memory: {:.1f} GB (Sufficient)", totalGB);
        } else {
            LOG_WARNING("System memory: {:.1f} GB (8+ GB recommended)", totalGB);
        }
    }
    
    // Check for GPU (basic check)
    DISPLAY_DEVICE dd;
    dd.cb = sizeof(DISPLAY_DEVICE);
    if (EnumDisplayDevices(NULL, 0, &dd, 0)) {
        LOG_INFO("Display adapter: {}", reinterpret_cast<const char*>(dd.DeviceString));
    }
#endif
    
    // Check CPU core count
    unsigned int numCores = std::thread::hardware_concurrency();
    if (numCores >= 4) {
        LOG_SUCCESS("CPU cores: {} (Sufficient)", numCores);
    } else {
        LOG_WARNING("CPU cores: {} (4+ cores recommended)", numCores);
    }
    
    return allRequirementsMet;
}

/**
 * @brief Handle unhandled exceptions
 */
void HandleUnhandledException() {
    try {
        std::rethrow_exception(std::current_exception());
    }
    catch (const std::bad_alloc& e) {
        LOG_ERROR("Memory allocation failed: {}", e.what());
        LOG_ERROR("The system may be running low on memory.");
    }
    catch (const std::runtime_error& e) {
        LOG_ERROR("Runtime error: {}", e.what());
    }
    catch (const std::logic_error& e) {
        LOG_ERROR("Logic error: {}", e.what());
    }
    catch (const std::exception& e) {
        LOG_ERROR("Unhandled exception: {}", e.what());
    }
    catch (...) {
        LOG_ERROR("Unknown exception occurred");
    }
    
    LOG_ERROR("Application will terminate. Please check the logs for details.");
    
#ifdef _WIN32
    LOG_INFO("Press any key to exit...");
    _getch();
#endif
}

/**
 * @brief Application entry point
 */
int main(int argc, char* argv[]) {
    // Set up global exception handler
    std::set_terminate(HandleUnhandledException);
    
    try {
        // Configure platform-specific settings
        ConfigureWindowsConsole();
        
        // Initialize logging
        InitializeLogging();
        
        // Display banner
        DisplayBanner();
        
        // Check system requirements
        if (!CheckSystemRequirements()) {
            LOG_WARNING("Some system requirements are not optimal. Performance may be affected.");
        }
        
        LOG_INFO("Initializing AI-Artworks application...");
        
        // Parse command line arguments
        bool verboseMode = false;
        std::string modelPath = "models/dolphin-mixtral.gguf";
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--verbose" || arg == "-v") {
                verboseMode = true;
            } else if (arg == "--model" || arg == "-m") {
                if (i + 1 < argc) {
                    modelPath = argv[++i];
                }
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "AI-Artworks Usage:\n"
                         << "  --verbose, -v    Enable verbose output\n"
                         << "  --model, -m      Specify model path (default: models/dolphin-mixtral.gguf)\n"
                         << "  --help, -h       Show this help message\n";
                return 0;
            }
        }
        
        if (verboseMode) {
            LOG_INFO("Verbose mode enabled");
#ifdef HAS_SPDLOG
            spdlog::set_level(spdlog::level::debug);
#endif
        }
        
        // Initialize and run application
        {
            LOG_INFO("Creating application instance...");
            Application& app = Application::getInstance();
            
            LOG_INFO("Initializing application systems...");
            if (!app.initialize()) {
                LOG_ERROR("Failed to initialize application");
                return 1;
            }
            
            // Load AI model if specified
            if (!modelPath.empty()) {
                LOG_INFO("Loading AI model: {}", modelPath);
                try {
                    auto& aiProcessor = app.getAIProcessor();
                    if (!aiProcessor.loadModel(modelPath)) {
                        LOG_WARNING("Failed to load AI model. Some features may be unavailable.");
                    } else {
                        LOG_SUCCESS("AI model loaded successfully");
                        
                        // Test model with a simple prompt
                        std::string testPrompt = "Generate a description for a digital artwork featuring abstract geometric patterns.";
                        LOG_INFO("Testing AI model with prompt: '{}'", testPrompt);
                        
                        std::string result = aiProcessor.processText(testPrompt);
                        if (!result.empty()) {
                            LOG_SUCCESS("AI model test successful");
                            LOG_INFO("AI Response: {}", result);
                        }
                    }
                }
                catch (const std::exception& e) {
                    LOG_ERROR("Error loading AI model: {}", e.what());
                }
            }
            
            LOG_SUCCESS("Application initialized successfully");
            LOG_INFO("Starting main application loop...");
            
            // Run main application loop
            int result = app.run();
            
            LOG_INFO("Application loop ended with code: {}", result);
            LOG_INFO("Shutting down application...");
            
            app.shutdown();
            LOG_SUCCESS("Application shutdown complete");
            
            return result;
        }
    }
    catch (const std::bad_alloc& e) {
        LOG_ERROR("Memory allocation failed: {}", e.what());
        LOG_ERROR("The system may be running low on memory or the requested allocation is too large.");
        return 2;
    }
    catch (const std::runtime_error& e) {
        LOG_ERROR("Runtime error: {}", e.what());
        return 3;
    }
    catch (const std::exception& e) {
        LOG_ERROR("Fatal error: {}", e.what());
        return 4;
    }
    catch (...) {
        LOG_ERROR("Unknown fatal error occurred");
        return 5;
    }
}

// Windows-specific entry point
#ifdef _WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Allocate console for GUI applications
    if (AllocConsole()) {
        freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);
        freopen_s((FILE**)stderr, "CONOUT$", "w", stderr);
        freopen_s((FILE**)stdin, "CONIN$", "r", stdin);
    }
    
    // Parse command line
    int argc = 0;
    LPWSTR* argv_w = CommandLineToArgvW(GetCommandLineW(), &argc);
    
    // Convert to char*
    std::vector<std::string> args;
    std::vector<char*> argv;
    
    for (int i = 0; i < argc; ++i) {
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, argv_w[i], -1, NULL, 0, NULL, NULL);
        std::string str(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, argv_w[i], -1, &str[0], size_needed, NULL, NULL);
        str.resize(size_needed - 1); // Remove null terminator
        args.push_back(str);
        argv.push_back(&args.back()[0]);
    }
    
    LocalFree(argv_w);
    
    return main(argc, argv.data());
}
#endif