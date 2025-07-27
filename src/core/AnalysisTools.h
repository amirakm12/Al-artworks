#pragma once

#include <vector>
#include <string>
#include "UltimateToolEngine.h"

namespace AI_ARTWORKS {

class AnalysisTools {
public:
    // Data Analysis Tools
    static bool dataAnalyzer(const std::vector<std::string>& params);
    static bool statisticalAnalyzer(const std::vector<std::string>& params);
    static bool trendAnalyzer(const std::vector<std::string>& params);
    static bool patternRecognizer(const std::vector<std::string>& params);
    static bool correlationAnalyzer(const std::vector<std::string>& params);
    static bool regressionAnalyzer(const std::vector<std::string>& params);
    static bool clusteringAnalyzer(const std::vector<std::string>& params);
    static bool classificationAnalyzer(const std::vector<std::string>& params);
    static bool timeSeriesAnalyzer(const std::vector<std::string>& params);
    static bool anomalyDetector(const std::vector<std::string>& params);

    // Advanced Analytics Tools
    static bool predictiveAnalyzer(const std::vector<std::string>& params);
    static bool prescriptiveAnalyzer(const std::vector<std::string>& params);
    static bool diagnosticAnalyzer(const std::vector<std::string>& params);
    static bool descriptiveAnalyzer(const std::vector<std::string>& params);
    static bool exploratoryAnalyzer(const std::vector<std::string>& params);
    static bool confirmatoryAnalyzer(const std::vector<std::string>& params);
    static bool inferentialAnalyzer(const std::vector<std::string>& params);
    static bool causalAnalyzer(const std::vector<std::string>& params);
    static bool bayesianAnalyzer(const std::vector<std::string>& params);
    static bool frequentistAnalyzer(const std::vector<std::string>& params);

    // Visualization Tools
    static bool chartGenerator(const std::vector<std::string>& params);
    static bool graphVisualizer(const std::vector<std::string>& params);
    static bool dashboardCreator(const std::vector<std::string>& params);
    static bool reportGenerator(const std::vector<std::string>& params);
    static bool infographicCreator(const std::vector<std::string>& params);
    static bool plotGenerator(const std::vector<std::string>& params);
    static bool histogramCreator(const std::vector<std::string>& params);
    static bool scatterPlotGenerator(const std::vector<std::string>& params);
    static bool heatmapGenerator(const std::vector<std::string>& params);
    static bool boxPlotCreator(const std::vector<std::string>& params);

    // Machine Learning Analysis Tools
    static bool modelEvaluator(const std::vector<std::string>& params);
    static bool featureAnalyzer(const std::vector<std::string>& params);
    static bool performanceAnalyzer(const std::vector<std::string>& params);
    static bool accuracyAnalyzer(const std::vector<std::string>& params);
    static bool precisionAnalyzer(const std::vector<std::string>& params);
    static bool recallAnalyzer(const std::vector<std::string>& params);
    static bool f1ScoreAnalyzer(const std::vector<std::string>& params);
    static bool rocAnalyzer(const std::vector<std::string>& params);
    static bool confusionMatrixAnalyzer(const std::vector<std::string>& params);
    static bool crossValidationAnalyzer(const std::vector<std::string>& params);

    // Business Intelligence Tools
    static bool kpiAnalyzer(const std::vector<std::string>& params);
    static bool metricAnalyzer(const std::vector<std::string>& params);
    static bool benchmarkAnalyzer(const std::vector<std::string>& params);
    static bool competitiveAnalyzer(const std::vector<std::string>& params);
    static bool marketAnalyzer(const std::vector<std::string>& params);
    static bool customerAnalyzer(const std::vector<std::string>& params);
    static bool productAnalyzer(const std::vector<std::string>& params);
    static bool salesAnalyzer(const std::vector<std::string>& params);
    static bool revenueAnalyzer(const std::vector<std::string>& params);
    static bool profitAnalyzer(const std::vector<std::string>& params);

    // Tool Registration
    static void registerAllTools(UltimateToolEngine& engine);
};

} // namespace AI_ARTWORKS 