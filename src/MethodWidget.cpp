#include "MethodWidget.h"
#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QDoubleValidator>
#include <QIntValidator>

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
    
    QLineEdit *gsSearchDepth = new QLineEdit("5", this);
    gsSearchDepth->setValidator(new QIntValidator(1, 100, this));
    gsLayout->addRow("Search Depth (km):", gsSearchDepth);
    
    QLineEdit *gsMinMisfit = new QLineEdit("0.001", this);
    gsLayout->addRow("Min Misfit Threshold:", gsMinMisfit);
    
    globalParamStack->addWidget(gridSearchParams);
    
    // Simulated Annealing parameters
    simulatedAnnealingParams = new QWidget();
    QFormLayout *saLayout = new QFormLayout(simulatedAnnealingParams);
    
    QLineEdit *saInitTemp = new QLineEdit("1000.0", this);
    saInitTemp->setValidator(new QDoubleValidator(0.0, 1e6, 3, this));
    saLayout->addRow("Initial Temperature:", saInitTemp);
    
    QLineEdit *saFinalTemp = new QLineEdit("0.1", this);
    saFinalTemp->setValidator(new QDoubleValidator(0.0, 1000.0, 3, this));
    saLayout->addRow("Final Temperature:", saFinalTemp);
    
    QLineEdit *saCoolingRate = new QLineEdit("0.95", this);
    saCoolingRate->setValidator(new QDoubleValidator(0.5, 0.9999, 4, this));
    saLayout->addRow("Cooling Rate:", saCoolingRate);
    
    QLineEdit *saIterPerTemp = new QLineEdit("100", this);
    saIterPerTemp->setValidator(new QIntValidator(1, 10000, this));
    saLayout->addRow("Iterations per Temp:", saIterPerTemp);
    
    QLineEdit *saMaxIter = new QLineEdit("10000", this);
    saMaxIter->setValidator(new QIntValidator(100, 1000000, this));
    saLayout->addRow("Max Total Iterations:", saMaxIter);
    
    globalParamStack->addWidget(simulatedAnnealingParams);
    
    // Genetic Algorithm parameters
    geneticAlgorithmParams = new QWidget();
    QFormLayout *gaLayout = new QFormLayout(geneticAlgorithmParams);
    
    QLineEdit *gaPopSize = new QLineEdit("100", this);
    gaPopSize->setValidator(new QIntValidator(10, 10000, this));
    gaLayout->addRow("Population Size:", gaPopSize);
    
    QLineEdit *gaGenerations = new QLineEdit("200", this);
    gaGenerations->setValidator(new QIntValidator(10, 10000, this));
    gaLayout->addRow("Max Generations:", gaGenerations);
    
    QLineEdit *gaCrossoverRate = new QLineEdit("0.8", this);
    gaCrossoverRate->setValidator(new QDoubleValidator(0.0, 1.0, 3, this));
    gaLayout->addRow("Crossover Rate:", gaCrossoverRate);
    
    QLineEdit *gaMutationRate = new QLineEdit("0.1", this);
    gaMutationRate->setValidator(new QDoubleValidator(0.0, 1.0, 3, this));
    gaLayout->addRow("Mutation Rate:", gaMutationRate);
    
    QLineEdit *gaEliteSize = new QLineEdit("10", this);
    gaEliteSize->setValidator(new QIntValidator(0, 100, this));
    gaLayout->addRow("Elite Size:", gaEliteSize);
    
    QLineEdit *gaTournamentSize = new QLineEdit("5", this);
    gaTournamentSize->setValidator(new QIntValidator(2, 50, this));
    gaLayout->addRow("Tournament Size:", gaTournamentSize);
    
    globalParamStack->addWidget(geneticAlgorithmParams);
    
    QGroupBox *paramGroup = new QGroupBox("Parameter Metode", this);
    QVBoxLayout *paramLayout = new QVBoxLayout();
    paramLayout->addWidget(globalParamStack);
    paramGroup->setLayout(paramLayout);
    
    globalLayout->addWidget(paramGroup);
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
