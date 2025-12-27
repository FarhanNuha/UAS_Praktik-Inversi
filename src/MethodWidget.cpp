#include "MethodWidget.h"
#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QDoubleValidator>
#include <QIntValidator>
#include <QCheckBox>

MethodWidget::MethodWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

MethodWidget::~MethodWidget() {
}

void MethodWidget::setupUI() {
    QScrollArea *scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    
    QWidget *contentWidget = new QWidget();
    QVBoxLayout *mainLayout = new QVBoxLayout(contentWidget);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    
    // Approach selection
    QGroupBox *approachGroup = new QGroupBox("Pendekatan Inversi", this);
    QVBoxLayout *approachLayout = new QVBoxLayout();
    
    approachCombo = new QComboBox(this);
    approachCombo->addItem("Pendekatan Lokal");
    approachCombo->addItem("Pendekatan Global");
    connect(approachCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MethodWidget::onApproachChanged);
    approachLayout->addWidget(approachCombo);
    
    approachGroup->setLayout(approachLayout);
    mainLayout->addWidget(approachGroup);
    
    // Method stack (local vs global)
    methodStack = new QStackedWidget(this);
    
    createLocalMethodWidgets();
    createGlobalMethodWidgets();
    
    methodStack->addWidget(localWidget);
    methodStack->addWidget(globalWidget);
    
    mainLayout->addWidget(methodStack);
    mainLayout->addStretch();
    
    scrollArea->setWidget(contentWidget);
    
    QVBoxLayout *outerLayout = new QVBoxLayout(this);
    outerLayout->setContentsMargins(0, 0, 0, 0);
    outerLayout->addWidget(scrollArea);
}

void MethodWidget::createLocalMethodWidgets() {
    localWidget = new QWidget();
    QVBoxLayout *localLayout = new QVBoxLayout(localWidget);
    localLayout->setSpacing(10);
    
    // Local method selection
    QGroupBox *methodGroup = new QGroupBox("Metode Lokal", this);
    QVBoxLayout *methodLayout = new QVBoxLayout();
    
    localMethodCombo = new QComboBox(this);
    localMethodCombo->addItem("Gauss Newton");
    localMethodCombo->addItem("Steepest Descent");
    localMethodCombo->addItem("Levenberg Marquardt");
    connect(localMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MethodWidget::onLocalMethodChanged);
    methodLayout->addWidget(localMethodCombo);
    
    methodGroup->setLayout(methodLayout);
    localLayout->addWidget(methodGroup);
    
    // Parameter stack for local methods
    localParamStack = new QStackedWidget(this);
    
    // Gauss Newton parameters
    gaussNewtonParams = new QWidget();
    QFormLayout *gnLayout = new QFormLayout(gaussNewtonParams);
    
    QLineEdit *gnMaxIter = new QLineEdit("100", this);
    gnMaxIter->setValidator(new QIntValidator(1, 10000, this));
    gnLayout->addRow("Max Iterations:", gnMaxIter);
    
    QLineEdit *gnTolerance = new QLineEdit("1e-6", this);
    gnLayout->addRow("Tolerance:", gnTolerance);
    
    QLineEdit *gnDamping = new QLineEdit("0.001", this);
    gnDamping->setValidator(new QDoubleValidator(0.0, 1.0, 6, this));
    gnLayout->addRow("Damping Factor:", gnDamping);
    
    localParamStack->addWidget(gaussNewtonParams);
    
    // Steepest Descent parameters
    steepestDescentParams = new QWidget();
    QFormLayout *sdLayout = new QFormLayout(steepestDescentParams);
    
    QLineEdit *sdMaxIter = new QLineEdit("200", this);
    sdMaxIter->setValidator(new QIntValidator(1, 10000, this));
    sdLayout->addRow("Max Iterations:", sdMaxIter);
    
    QLineEdit *sdTolerance = new QLineEdit("1e-5", this);
    sdLayout->addRow("Tolerance:", sdTolerance);
    
    QLineEdit *sdStepSize = new QLineEdit("0.01", this);
    sdStepSize->setValidator(new QDoubleValidator(0.0, 1.0, 6, this));
    sdLayout->addRow("Step Size:", sdStepSize);
    
    QLineEdit *sdLineSearch = new QLineEdit("10", this);
    sdLineSearch->setValidator(new QIntValidator(1, 100, this));
    sdLayout->addRow("Line Search Iter:", sdLineSearch);
    
    localParamStack->addWidget(steepestDescentParams);
    
    // Levenberg Marquardt parameters
    levenbergMarquadtParams = new QWidget();
    QFormLayout *lmLayout = new QFormLayout(levenbergMarquadtParams);
    
    QLineEdit *lmMaxIter = new QLineEdit("100", this);
    lmMaxIter->setValidator(new QIntValidator(1, 10000, this));
    lmLayout->addRow("Max Iterations:", lmMaxIter);
    
    QLineEdit *lmTolerance = new QLineEdit("1e-6", this);
    lmLayout->addRow("Tolerance:", lmTolerance);
    
    QLineEdit *lmLambda = new QLineEdit("0.01", this);
    lmLambda->setValidator(new QDoubleValidator(0.0, 1000.0, 6, this));
    lmLayout->addRow("Lambda (λ):", lmLambda);
    
    QLineEdit *lmLambdaUp = new QLineEdit("10.0", this);
    lmLambdaUp->setValidator(new QDoubleValidator(1.0, 100.0, 3, this));
    lmLayout->addRow("Lambda Up Factor:", lmLambdaUp);
    
    QLineEdit *lmLambdaDown = new QLineEdit("0.1", this);
    lmLambdaDown->setValidator(new QDoubleValidator(0.01, 1.0, 3, this));
    lmLayout->addRow("Lambda Down Factor:", lmLambdaDown);
    
    localParamStack->addWidget(levenbergMarquadtParams);
    
    QGroupBox *paramGroup = new QGroupBox("Parameter Metode", this);
    QVBoxLayout *paramLayout = new QVBoxLayout();
    paramLayout->addWidget(localParamStack);
    paramGroup->setLayout(paramLayout);
    
    localLayout->addWidget(paramGroup);
}

void MethodWidget::createGlobalMethodWidgets() {
    globalWidget = new QWidget();
    QVBoxLayout *globalLayout = new QVBoxLayout(globalWidget);
    globalLayout->setSpacing(10);
    
    // Global method selection
    QGroupBox *methodGroup = new QGroupBox("Metode Global", this);
    QVBoxLayout *methodLayout = new QVBoxLayout();
    
    globalMethodCombo = new QComboBox(this);
    globalMethodCombo->addItem("Grid Search");
    globalMethodCombo->addItem("Simulated Annealing");
    globalMethodCombo->addItem("Algoritma Genetika");
    connect(globalMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MethodWidget::onGlobalMethodChanged);
    methodLayout->addWidget(globalMethodCombo);
    
    methodGroup->setLayout(methodLayout);
    globalLayout->addWidget(methodGroup);
    
    // Parameter stack for global methods
    globalParamStack = new QStackedWidget(this);
    
    // Grid Search parameters
    gridSearchParams = new QWidget();
    QFormLayout *gsLayout = new QFormLayout(gridSearchParams);
    
    QLabel *gsNote = new QLabel("<i>Grid didefinisikan di 'Calculating Condition'</i>", this);
    gsNote->setWordWrap(true);
    gsLayout->addRow(gsNote);
    
    QLabel *gsInfo = new QLabel(
        "<b>Grid Search:</b><br>"
        "Pencarian exhaustive pada seluruh grid.<br>"
        "Perhitungan berhenti setelah semua grid dievaluasi.",
        this
    );
    gsInfo->setWordWrap(true);
    gsInfo->setStyleSheet("QLabel { background-color: #e3f2fd; padding: 8px; border-radius: 3px; }");
    gsLayout->addRow(gsInfo);
    
    globalParamStack->addWidget(gridSearchParams);
    
    // Simulated Annealing parameters (with variants)
    createSimulatedAnnealingWidgets();
    globalParamStack->addWidget(simulatedAnnealingParams);
    
    // Genetic Algorithm parameters (with variants)
    createGeneticAlgorithmWidgets();
    globalParamStack->addWidget(geneticAlgorithmParams);
    
    QGroupBox *paramGroup = new QGroupBox("Parameter Metode", this);
    QVBoxLayout *paramLayout = new QVBoxLayout();
    paramLayout->addWidget(globalParamStack);
    paramGroup->setLayout(paramLayout);
    
    globalLayout->addWidget(paramGroup);
}

void MethodWidget::createSimulatedAnnealingWidgets() {
    simulatedAnnealingParams = new QWidget();
    QVBoxLayout *saMainLayout = new QVBoxLayout(simulatedAnnealingParams);
    
    // SA Variant Selection
    QGroupBox *variantGroup = new QGroupBox("Simulated Annealing Variant", this);
    QVBoxLayout *variantLayout = new QVBoxLayout();
    
    saVariantCombo = new QComboBox(this);
    saVariantCombo->addItem("Simple Simulated Annealing");
    saVariantCombo->addItem("Metropolis Simulated Annealing");
    saVariantCombo->addItem("Cauchy Simulated Annealing");
    connect(saVariantCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MethodWidget::onSAVariantChanged);
    variantLayout->addWidget(saVariantCombo);
    
    variantGroup->setLayout(variantLayout);
    saMainLayout->addWidget(variantGroup);
    
    // SA Variant Parameters Stack
    saVariantStack = new QStackedWidget(this);
    
    // === Simple SA Parameters ===
    simpleSAParams = new QWidget();
    QFormLayout *simpleSALayout = new QFormLayout(simpleSAParams);
    
    QLineEdit *simpleInitTemp = new QLineEdit("1000.0", this);
    simpleInitTemp->setValidator(new QDoubleValidator(0.0, 1e6, 3, this));
    simpleSALayout->addRow("Initial Temperature (T₀):", simpleInitTemp);
    
    QLineEdit *simpleFinalTemp = new QLineEdit("0.1", this);
    simpleFinalTemp->setValidator(new QDoubleValidator(0.0, 1000.0, 3, this));
    simpleSALayout->addRow("Final Temperature (Tf):", simpleFinalTemp);
    
    QComboBox *simpleCoolingScheme = new QComboBox(this);
    simpleCoolingScheme->addItem("Exponential: T(k) = T₀ × α^k");
    simpleCoolingScheme->addItem("Linear: T(k) = T₀ - α × k");
    simpleCoolingScheme->addItem("Logarithmic: T(k) = T₀ / (1 + α × log(1+k))");
    simpleCoolingScheme->addItem("Inverse: T(k) = T₀ / (1 + α × k)");
    simpleSALayout->addRow("Cooling Schedule:", simpleCoolingScheme);
    
    QLineEdit *simpleAlpha = new QLineEdit("0.95", this);
    simpleAlpha->setValidator(new QDoubleValidator(0.0, 1.0, 6, this));
    simpleSALayout->addRow("Alpha (α):", simpleAlpha);
    
    QLineEdit *simpleIterPerTemp = new QLineEdit("100", this);
    simpleIterPerTemp->setValidator(new QIntValidator(1, 10000, this));
    simpleSALayout->addRow("Iterations per Temp:", simpleIterPerTemp);
    
    QLineEdit *simpleMaxIter = new QLineEdit("10000", this);
    simpleMaxIter->setValidator(new QIntValidator(100, 1000000, this));
    simpleSALayout->addRow("Max Total Iterations:", simpleMaxIter);
    
    QLabel *simpleNote = new QLabel(
        "<i>Simple SA: Standar SA dengan penurunan temperatur sederhana</i>",
        this
    );
    simpleNote->setWordWrap(true);
    simpleNote->setStyleSheet("QLabel { color: #666; padding: 5px; }");
    simpleSALayout->addRow(simpleNote);
    
    saVariantStack->addWidget(simpleSAParams);
    
    // === Metropolis SA Parameters ===
    metropolisSAParams = new QWidget();
    QFormLayout *metropolisSALayout = new QFormLayout(metropolisSAParams);
    
    QLineEdit *metroInitTemp = new QLineEdit("1500.0", this);
    metroInitTemp->setValidator(new QDoubleValidator(0.0, 1e6, 3, this));
    metropolisSALayout->addRow("Initial Temperature (T₀):", metroInitTemp);
    
    QLineEdit *metroFinalTemp = new QLineEdit("0.05", this);
    metroFinalTemp->setValidator(new QDoubleValidator(0.0, 1000.0, 3, this));
    metropolisSALayout->addRow("Final Temperature (Tf):", metroFinalTemp);
    
    QComboBox *metropolisCoolingScheme = new QComboBox(this);
    metropolisCoolingScheme->addItem("Exponential: T(k) = T₀ × α^k");
    metropolisCoolingScheme->addItem("Linear: T(k) = T₀ - α × k");
    metropolisCoolingScheme->addItem("Logarithmic: T(k) = T₀ / (1 + α × log(1+k))");
    metropolisCoolingScheme->addItem("Adaptive: T(k) = T₀ × (1 - k/maxIter)^α");
    metropolisSALayout->addRow("Cooling Schedule:", metropolisCoolingScheme);
    
    QLineEdit *metroAlpha = new QLineEdit("0.90", this);
    metroAlpha->setValidator(new QDoubleValidator(0.0, 1.0, 6, this));
    metropolisSALayout->addRow("Alpha (α):", metroAlpha);
    
    QLineEdit *metroMarkovChain = new QLineEdit("150", this);
    metroMarkovChain->setValidator(new QIntValidator(1, 10000, this));
    metropolisSALayout->addRow("Markov Chain Length:", metroMarkovChain);
    
    QLineEdit *metroAcceptanceRatio = new QLineEdit("0.6", this);
    metroAcceptanceRatio->setValidator(new QDoubleValidator(0.0, 1.0, 3, this));
    metropolisSALayout->addRow("Target Acceptance Ratio:", metroAcceptanceRatio);
    
    QLineEdit *metroMaxIter = new QLineEdit("15000", this);
    metroMaxIter->setValidator(new QIntValidator(100, 1000000, this));
    metropolisSALayout->addRow("Max Total Iterations:", metroMaxIter);
    
    QLabel *metroNote = new QLabel(
        "<i>Metropolis SA: Menggunakan kriteria Metropolis untuk acceptance probability<br>"
        "P(accept) = exp(-ΔE/T) if ΔE > 0</i>",
        this
    );
    metroNote->setWordWrap(true);
    metroNote->setStyleSheet("QLabel { color: #666; padding: 5px; }");
    metropolisSALayout->addRow(metroNote);
    
    saVariantStack->addWidget(metropolisSAParams);
    
    // === Cauchy SA Parameters ===
    cauchySAParams = new QWidget();
    QFormLayout *cauchySALayout = new QFormLayout(cauchySAParams);
    
    QLineEdit *cauchyInitTemp = new QLineEdit("2000.0", this);
    cauchyInitTemp->setValidator(new QDoubleValidator(0.0, 1e6, 3, this));
    cauchySALayout->addRow("Initial Temperature (T₀):", cauchyInitTemp);
    
    QLineEdit *cauchyFinalTemp = new QLineEdit("0.01", this);
    cauchyFinalTemp->setValidator(new QDoubleValidator(0.0, 1000.0, 3, this));
    cauchySALayout->addRow("Final Temperature (Tf):", cauchyFinalTemp);
    
    QComboBox *cauchyCoolingScheme = new QComboBox(this);
    cauchyCoolingScheme->addItem("Cauchy: T(k) = T₀ / (1 + k)");
    cauchyCoolingScheme->addItem("Fast Cauchy: T(k) = T₀ / (1 + α × k)");
    cauchyCoolingScheme->addItem("Very Fast: T(k) = T₀ × exp(-α × k^(1/D))");
    cauchySALayout->addRow("Cooling Schedule:", cauchyCoolingScheme);
    
    QLineEdit *cauchyAlpha = new QLineEdit("1.0", this);
    cauchyAlpha->setValidator(new QDoubleValidator(0.1, 10.0, 6, this));
    cauchySALayout->addRow("Alpha (α):", cauchyAlpha);
    
    QLineEdit *cauchyDimension = new QLineEdit("3", this);
    cauchyDimension->setValidator(new QIntValidator(1, 100, this));
    cauchySALayout->addRow("Problem Dimension (D):", cauchyDimension);
    
    QLineEdit *cauchyStepSize = new QLineEdit("1.0", this);
    cauchyStepSize->setValidator(new QDoubleValidator(0.001, 100.0, 3, this));
    cauchySALayout->addRow("Initial Step Size:", cauchyStepSize);
    
    QLineEdit *cauchyIterPerTemp = new QLineEdit("200", this);
    cauchyIterPerTemp->setValidator(new QIntValidator(1, 10000, this));
    cauchySALayout->addRow("Iterations per Temp:", cauchyIterPerTemp);
    
    QLineEdit *cauchyMaxIter = new QLineEdit("20000", this);
    cauchyMaxIter->setValidator(new QIntValidator(100, 1000000, this));
    cauchySALayout->addRow("Max Total Iterations:", cauchyMaxIter);
    
    QLabel *cauchyNote = new QLabel(
        "<i>Cauchy SA: Menggunakan distribusi Cauchy untuk langkah pencarian<br>"
        "Cooling lebih lambat, cocok untuk landscape kompleks</i>",
        this
    );
    cauchyNote->setWordWrap(true);
    cauchyNote->setStyleSheet("QLabel { color: #666; padding: 5px; }");
    cauchySALayout->addRow(cauchyNote);
    
    saVariantStack->addWidget(cauchySAParams);
    
    // Add variant stack to main layout
    QGroupBox *variantParamGroup = new QGroupBox("Variant Parameters", this);
    QVBoxLayout *variantParamLayout = new QVBoxLayout();
    variantParamLayout->addWidget(saVariantStack);
    variantParamGroup->setLayout(variantParamLayout);
    
    saMainLayout->addWidget(variantParamGroup);
}

void MethodWidget::onApproachChanged(int index) {
    methodStack->setCurrentIndex(index);
}

void MethodWidget::onLocalMethodChanged(int index) {
    localParamStack->setCurrentIndex(index);
}

void MethodWidget::onGlobalMethodChanged(int index) {
    globalParamStack->setCurrentIndex(index);
}

void MethodWidget::onSAVariantChanged(int index) {
    saVariantStack->setCurrentIndex(index);
}

void MethodWidget::onGAVariantChanged(int index) {
    gaVariantStack->setCurrentIndex(index);
}

void MethodWidget::createGeneticAlgorithmWidgets() {
    geneticAlgorithmParams = new QWidget();
    QVBoxLayout *gaMainLayout = new QVBoxLayout(geneticAlgorithmParams);
    
    // Real-coded toggle
    gaRealCodedCheck = new QCheckBox("Real-Coded Genetic Algorithm", this);
    gaRealCodedCheck->setChecked(true);
    gaRealCodedCheck->setStyleSheet("QCheckBox { font-weight: bold; padding: 5px; }");
    gaMainLayout->addWidget(gaRealCodedCheck);
    
    QLabel *realCodedNote = new QLabel(
        "<i>Real-coded: Menggunakan representasi bilangan real (kontinyu)<br>"
        "Binary-coded: Menggunakan representasi biner (0/1)</i>",
        this
    );
    realCodedNote->setWordWrap(true);
    realCodedNote->setStyleSheet("QLabel { color: #666; padding: 5px; }");
    gaMainLayout->addWidget(realCodedNote);
    
    // GA Variant Selection
    QGroupBox *variantGroup = new QGroupBox("Genetic Algorithm Variant", this);
    QVBoxLayout *variantLayout = new QVBoxLayout();
    
    gaVariantCombo = new QComboBox(this);
    gaVariantCombo->addItem("Algoritma Genetika Biasa");
    gaVariantCombo->addItem("Steady State Algoritma Genetika");
    gaVariantCombo->addItem("Strength Pareto Evolutionary Algorithm 2 (SPEA2)");
    connect(gaVariantCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MethodWidget::onGAVariantChanged);
    variantLayout->addWidget(gaVariantCombo);
    
    variantGroup->setLayout(variantLayout);
    gaMainLayout->addWidget(variantGroup);
    
    // GA Variant Parameters Stack
    gaVariantStack = new QStackedWidget(this);
    
    // === Standard GA Parameters ===
    standardGAParams = new QWidget();
    QFormLayout *standardGALayout = new QFormLayout(standardGAParams);
    
    QLineEdit *stdPopSize = new QLineEdit("100", this);
    stdPopSize->setValidator(new QIntValidator(10, 10000, this));
    standardGALayout->addRow("Population Size:", stdPopSize);
    
    QLineEdit *stdGenerations = new QLineEdit("200", this);
    stdGenerations->setValidator(new QIntValidator(10, 10000, this));
    standardGALayout->addRow("Max Generations:", stdGenerations);
    
    QLineEdit *stdCrossoverRate = new QLineEdit("0.8", this);
    stdCrossoverRate->setValidator(new QDoubleValidator(0.0, 1.0, 3, this));
    standardGALayout->addRow("Crossover Rate:", stdCrossoverRate);
    
    QLineEdit *stdMutationRate = new QLineEdit("0.1", this);
    stdMutationRate->setValidator(new QDoubleValidator(0.0, 1.0, 3, this));
    standardGALayout->addRow("Mutation Rate:", stdMutationRate);
    
    QLineEdit *stdEliteSize = new QLineEdit("10", this);
    stdEliteSize->setValidator(new QIntValidator(0, 100, this));
    standardGALayout->addRow("Elite Size:", stdEliteSize);
    
    QComboBox *stdSelectionMethod = new QComboBox(this);
    stdSelectionMethod->addItem("Roulette Wheel Selection");
    stdSelectionMethod->addItem("Tournament Selection");
    stdSelectionMethod->addItem("Rank Selection");
    standardGALayout->addRow("Selection Method:", stdSelectionMethod);
    
    QLineEdit *stdTournamentSize = new QLineEdit("5", this);
    stdTournamentSize->setValidator(new QIntValidator(2, 50, this));
    standardGALayout->addRow("Tournament Size:", stdTournamentSize);
    
    QLabel *stdNote = new QLabel(
        "<i>Standard GA: Generational replacement, seluruh populasi diganti setiap generasi</i>",
        this
    );
    stdNote->setWordWrap(true);
    stdNote->setStyleSheet("QLabel { color: #666; padding: 5px; }");
    standardGALayout->addRow(stdNote);
    
    gaVariantStack->addWidget(standardGAParams);
    
    // === Steady State GA Parameters ===
    steadyStateGAParams = new QWidget();
    QFormLayout *ssGALayout = new QFormLayout(steadyStateGAParams);
    
    QLineEdit *ssPopSize = new QLineEdit("100", this);
    ssPopSize->setValidator(new QIntValidator(10, 10000, this));
    ssGALayout->addRow("Population Size:", ssPopSize);
    
    QLineEdit *ssMaxIterations = new QLineEdit("10000", this);
    ssMaxIterations->setValidator(new QIntValidator(100, 1000000, this));
    ssGALayout->addRow("Max Iterations:", ssMaxIterations);
    
    QLineEdit *ssOffspringSize = new QLineEdit("2", this);
    ssOffspringSize->setValidator(new QIntValidator(1, 20, this));
    ssGALayout->addRow("Offspring per Iteration:", ssOffspringSize);
    
    QLineEdit *ssCrossoverRate = new QLineEdit("0.9", this);
    ssCrossoverRate->setValidator(new QDoubleValidator(0.0, 1.0, 3, this));
    ssGALayout->addRow("Crossover Rate:", ssCrossoverRate);
    
    QLineEdit *ssMutationRate = new QLineEdit("0.05", this);
    ssMutationRate->setValidator(new QDoubleValidator(0.0, 1.0, 3, this));
    ssGALayout->addRow("Mutation Rate:", ssMutationRate);
    
    QComboBox *ssReplacementStrategy = new QComboBox(this);
    ssReplacementStrategy->addItem("Replace Worst");
    ssReplacementStrategy->addItem("Replace Random");
    ssReplacementStrategy->addItem("Replace Parent");
    ssGALayout->addRow("Replacement Strategy:", ssReplacementStrategy);
    
    QLineEdit *ssTournamentSize = new QLineEdit("3", this);
    ssTournamentSize->setValidator(new QIntValidator(2, 20, this));
    ssGALayout->addRow("Tournament Size:", ssTournamentSize);
    
    QLabel *ssNote = new QLabel(
        "<i>Steady State GA: Hanya beberapa individu diganti per iterasi,<br>"
        "konvergensi lebih cepat, diversity lebih terjaga</i>",
        this
    );
    ssNote->setWordWrap(true);
    ssNote->setStyleSheet("QLabel { color: #666; padding: 5px; }");
    ssGALayout->addRow(ssNote);
    
    gaVariantStack->addWidget(steadyStateGAParams);
    
    // === SPEA2 Parameters ===
    spea2Params = new QWidget();
    QFormLayout *spea2Layout = new QFormLayout(spea2Params);
    
    QLineEdit *spea2PopSize = new QLineEdit("100", this);
    spea2PopSize->setValidator(new QIntValidator(10, 10000, this));
    spea2Layout->addRow("Population Size:", spea2PopSize);
    
    QLineEdit *spea2ArchiveSize = new QLineEdit("100", this);
    spea2ArchiveSize->setValidator(new QIntValidator(10, 10000, this));
    spea2Layout->addRow("Archive Size:", spea2ArchiveSize);
    
    QLineEdit *spea2Generations = new QLineEdit("250", this);
    spea2Generations->setValidator(new QIntValidator(10, 10000, this));
    spea2Layout->addRow("Max Generations:", spea2Generations);
    
    QLineEdit *spea2CrossoverRate = new QLineEdit("0.9", this);
    spea2CrossoverRate->setValidator(new QDoubleValidator(0.0, 1.0, 3, this));
    spea2Layout->addRow("Crossover Rate:", spea2CrossoverRate);
    
    QLineEdit *spea2MutationRate = new QLineEdit("0.1", this);
    spea2MutationRate->setValidator(new QDoubleValidator(0.0, 1.0, 3, this));
    spea2Layout->addRow("Mutation Rate:", spea2MutationRate);
    
    QLineEdit *spea2KNearest = new QLineEdit("5", this);
    spea2KNearest->setValidator(new QIntValidator(1, 50, this));
    spea2Layout->addRow("K-Nearest Neighbors:", spea2KNearest);
    
    QLabel *spea2Note = new QLabel(
        "<i>SPEA2: Multi-objective optimization dengan Pareto front,<br>"
        "menggunakan strength fitness dan density estimation.<br>"
        "Cocok untuk masalah dengan multiple objectives.</i>",
        this
    );
    spea2Note->setWordWrap(true);
    spea2Note->setStyleSheet("QLabel { color: #666; padding: 5px; }");
    spea2Layout->addRow(spea2Note);
    
    gaVariantStack->addWidget(spea2Params);
    
    // Add variant stack to main layout
    QGroupBox *variantParamGroup = new QGroupBox("Variant Parameters", this);
    QVBoxLayout *variantParamLayout = new QVBoxLayout();
    variantParamLayout->addWidget(gaVariantStack);
    variantParamGroup->setLayout(variantParamLayout);
    
    gaMainLayout->addWidget(variantParamGroup);
}

QString MethodWidget::getApproach() const {
    return approachCombo->currentText();
}

QString MethodWidget::getMethod() const {
    if (approachCombo->currentIndex() == 0) {
        return localMethodCombo->currentText();
    } else {
        return globalMethodCombo->currentText();
    }
}

QString MethodWidget::getGlobalSubMethod() const {
    if (approachCombo->currentIndex() == 1 && globalMethodCombo->currentIndex() == 1) {
        // Simulated Annealing selected
        return saVariantCombo->currentText();
    }
    return "";
}
