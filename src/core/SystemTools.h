#pragma once

#include <vector>
#include <string>
#include "UltimateToolEngine.h"

namespace AI_ARTWORKS {

class SystemTools {
public:
    // File Management Tools
    static bool fileAnalyzer(const std::vector<std::string>& params);
    static bool fileOptimizer(const std::vector<std::string>& params);
    static bool fileEncryptor(const std::vector<std::string>& params);
    static bool fileCompressor(const std::vector<std::string>& params);
    static bool fileSynchronizer(const std::vector<std::string>& params);
    static bool fileValidator(const std::vector<std::string>& params);
    static bool fileConverter(const std::vector<std::string>& params);
    static bool fileIndexer(const std::vector<std::string>& params);
    static bool fileBackup(const std::vector<std::string>& params);
    static bool fileRecovery(const std::vector<std::string>& params);

    // Process Management Tools
    static bool processMonitor(const std::vector<std::string>& params);
    static bool processOptimizer(const std::vector<std::string>& params);
    static bool processKiller(const std::vector<std::string>& params);
    static bool processScheduler(const std::vector<std::string>& params);
    static bool processProfiler(const std::vector<std::string>& params);
    static bool processAnalyzer(const std::vector<std::string>& params);
    static bool processController(const std::vector<std::string>& params);
    static bool processBalancer(const std::vector<std::string>& params);
    static bool processRecovery(const std::vector<std::string>& params);
    static bool processSecurity(const std::vector<std::string>& params);

    // System Management Tools
    static bool systemMonitor(const std::vector<std::string>& params);
    static bool performanceAnalyzer(const std::vector<std::string>& params);
    static bool resourceTracker(const std::vector<std::string>& params);
    static bool systemDiagnostics(const std::vector<std::string>& params);
    static bool healthChecker(const std::vector<std::string>& params);
    static bool systemProfiler(const std::vector<std::string>& params);
    static bool systemPredictor(const std::vector<std::string>& params);
    static bool systemRepair(const std::vector<std::string>& params);
    static bool systemUpgrader(const std::vector<std::string>& params);

    // Optimization Tools
    static bool memoryOptimizer(const std::vector<std::string>& params);
    static bool cpuOptimizer(const std::vector<std::string>& params);
    static bool diskOptimizer(const std::vector<std::string>& params);
    static bool networkOptimizer(const std::vector<std::string>& params);
    static bool powerOptimizer(const std::vector<std::string>& params);
    static bool cacheOptimizer(const std::vector<std::string>& params);
    static bool threadOptimizer(const std::vector<std::string>& params);
    static bool bufferOptimizer(const std::vector<std::string>& params);
    static bool queueOptimizer(const std::vector<std::string>& params);
    static bool poolOptimizer(const std::vector<std::string>& params);

    // System Maintenance Tools
    static bool registryCleaner(const std::vector<std::string>& params);
    static bool serviceManager(const std::vector<std::string>& params);
    static bool driverUpdater(const std::vector<std::string>& params);
    static bool systemRestorer(const std::vector<std::string>& params);
    static bool systemBackup(const std::vector<std::string>& params);

    // Utility Functions
    static bool isValidFile(const std::string& path);
    static bool isValidDirectory(const std::string& path);
    static bool killProcess(const std::string& process_name);
    static bool isProcessRunning(const std::string& process_name);
    static bool optimizeMemory();
    static bool optimizeCPU();
    static bool optimizeDisk();
    static bool optimizeNetwork();
    static bool optimizePower();

    // Tool Registration
    static void registerAllTools(UltimateToolEngine& engine);

private:
    static uint64_t getFileSize(const std::string& path);
    static std::string getFileExtension(const std::string& filename);
    static std::string getFileHash(const std::string& path);
};

} // namespace AI_ARTWORKS 