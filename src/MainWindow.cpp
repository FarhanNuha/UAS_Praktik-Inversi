#include "MainWindow.h"
#include "MapViewer2D.h"
#include "MapViewer3D.h"
#include "CalculatingConditionWidget.h"
#include "MethodWidget.h"
#include "VelocityModelWidget.h"
#include "DataInputWidget.h"
#include "ResultWidget.h"

#include <QApplication>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <cmath>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setupUI();
    resize(1600, 1000);
    setWindowTitle("Earthquake Hypocenter Inversion Software");
}

MainWindow::~MainWindow() {
}

void MainWindow::setupUI() {
    createMenuBar();
    createStatusBar();
    createCentralWidget();
    
    updateStatusBar("Ready");
}

void MainWindow::createMenuBar() {
    QMenuBar *menuBar = new QMenuBar(this);
    setMenuBar(menuBar);
    
    QMenu *fileMenu = menuBar->addMenu("&File");
    
    newAction = new QAction("&New", this);
    newAction->setShortcut(QKeySequence::New);
    connect(newAction, &QAction::triggered, this, &MainWindow::onNewFile);
    fileMenu->addAction(newAction);
    
    openAction = new QAction("&Open", this);
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, &MainWindow::onOpenFile);
    fileMenu->addAction(openAction);
    
    saveAction = new QAction("&Save", this);
    saveAction->setShortcut(QKeySequence::Save);
    connect(saveAction, &QAction::triggered, this, &MainWindow::onSaveFile);
    fileMenu->addAction(saveAction);
    
    fileMenu->addSeparator();
    
    exitAction = new QAction("E&xit", this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &MainWindow::onExit);
    fileMenu->addAction(exitAction);
    
    QMenu *editMenu = menuBar->addMenu("&Edit");
    QAction *preferencesAction = new QAction("&Preferences", this);
    editMenu->addAction(preferencesAction);
    
    QMenu *settingsMenu = menuBar->addMenu("&Settings");
    QAction *configAction = new QAction("&Configuration", this);
    settingsMenu->addAction(configAction);
    
    QMenu *helpMenu = menuBar->addMenu("&Help");
    
    aboutAction = new QAction("&About", this);
    connect(aboutAction, &QAction::triggered, this, &MainWindow::onAbout);
    helpMenu->addAction(aboutAction);
}

void MainWindow::createStatusBar() {
    QStatusBar *statusBar = new QStatusBar(this);
    setStatusBar(statusBar);
    
    statusLabel = new QLabel("Ready", this);
    statusBar->addWidget(statusLabel, 1);
    
    authorLabel = new QLabel("Farhan Nuha Zhofiro / 31.22.0006", this);
    statusBar->addPermanentWidget(authorLabel);
}

void MainWindow::createCentralWidget() {
    mainSplitter = new QSplitter(Qt::Horizontal, this);
    
    // LEFT PANEL - Tab Widget
    leftTabWidget = new QTabWidget(this);
    
    mapViewer2D = new MapViewer2D(this);
    leftTabWidget->addTab(mapViewer2D, "2D View");
    
    mapViewer3D = new MapViewer3D(this);
    leftTabWidget->addTab(mapViewer3D, "3D View");
    
    resultWidget = new ResultWidget(this);
    leftTabWidget->addTab(resultWidget, "Hasil");
    
    mainSplitter->addWidget(leftTabWidget);
    
    // RIGHT PANEL - Vertical splitter
    QSplitter *rightSplitter = new QSplitter(Qt::Vertical, this);
    
    // Top right tab widget
    topRightTabWidget = new QTabWidget(this);
    
    calcConditionWidget = new CalculatingConditionWidget(this);
    topRightTabWidget->addTab(calcConditionWidget, "Calculating Condition");
    
    methodWidget = new MethodWidget(this);
    topRightTabWidget->addTab(methodWidget, "Metode");
    
    rightSplitter->addWidget(topRightTabWidget);
    
    // Bottom right tab widget
    bottomRightTabWidget = new QTabWidget(this);
    
    velocityModelWidget = new VelocityModelWidget(this);
    bottomRightTabWidget->addTab(velocityModelWidget, "Velocity Model");
    
    dataInputWidget = new DataInputWidget(this);
    bottomRightTabWidget->addTab(dataInputWidget, "Data Input");
    
    rightSplitter->addWidget(bottomRightTabWidget);
    
    rightSplitter->setStretchFactor(0, 3);
    rightSplitter->setStretchFactor(1, 2);
    
    mainSplitter->addWidget(rightSplitter);
    mainSplitter->setStretchFactor(0, 8);
    mainSplitter->setStretchFactor(1, 3);
    
    setCentralWidget(mainSplitter);
    
    // Connect signals for data flow
    
    // 1. Calculating Condition -> Maps & Velocity & Method & DataInput
    connect(calcConditionWidget, &CalculatingConditionWidget::conditionCommitted,
            mapViewer2D, &MapViewer2D::updateBoundary);
    connect(calcConditionWidget, &CalculatingConditionWidget::conditionCommitted,
            mapViewer3D, &MapViewer3D::updateBoundary);
    connect(calcConditionWidget, &CalculatingConditionWidget::conditionCommitted,
            velocityModelWidget, &VelocityModelWidget::setBoundary);
    connect(calcConditionWidget, &CalculatingConditionWidget::conditionCommitted,
            methodWidget, &MethodWidget::updateGridSize);
    connect(calcConditionWidget, &CalculatingConditionWidget::conditionCommitted,
            dataInputWidget, &DataInputWidget::setBoundaryData);
    
    // 2. Stations -> Maps
    connect(dataInputWidget, &DataInputWidget::stationsLoaded,
            mapViewer2D, &MapViewer2D::updateStations);
    connect(dataInputWidget, &DataInputWidget::stationsLoaded,
            mapViewer3D, &MapViewer3D::updateStations);
    
    // 3. Computation Complete -> Results
    connect(dataInputWidget, &DataInputWidget::computationComplete,
            this, [this](const QString &resultText, const HypocenterResult &result) {
        // Display text result
        resultWidget->clearResults();
        resultWidget->appendResult(resultText);
        
        // Display misfit plot
        if (!result.iterationNumbers.isEmpty() && !result.misfitHistory.isEmpty()) {
            resultWidget->setMisfitData(result.iterationNumbers, result.misfitHistory);
        } else {
            // Generate dummy data for visualization if not available
            QVector<double> dummyIter, dummyMisfit;
            for (int i = 0; i <= 10; ++i) {
                dummyIter.append(i);
                dummyMisfit.append(result.rms / (1.0 + i * 0.1));
            }
            resultWidget->setMisfitData(dummyIter, dummyMisfit);
        }
        
        // Get station data for visualization
        QVector<StationData> stations = dataInputWidget->getStationData();
        
        // Display 2D result - generate dummy contour if not available
        QVector<QPointF> contour = result.contour2D;
        if (contour.isEmpty()) {
            // Generate circular contour around result point
            for (int i = 0; i < 8; ++i) {
                double angle = 2 * M_PI * i / 8;
                double radius = 0.5;
                contour.append(QPointF(result.x + radius * cos(angle), 
                                      result.y + radius * sin(angle)));
            }
        }
        resultWidget->set2DResult(result.x, result.y, contour, stations);
        
        // Display 3D result with station data
        resultWidget->set3DResult(result.x, result.y, result.z, result.contour2D, stations);
        
        // Switch to Result tab
        leftTabWidget->setCurrentWidget(resultWidget);
        
        // Update status bar
        updateStatusBar(QString("Computation complete! Location: %1°, %2°, %3 km")
                       .arg(result.x, 0, 'f', 4)
                       .arg(result.y, 0, 'f', 4)
                       .arg(result.z, 0, 'f', 2));
    });
    
    // 4. Pass method data to DataInput when user explicitly commits selection
    connect(methodWidget, &MethodWidget::methodCommitted,
            this, [this](const QString &approach, const QString &method, bool useMC, int mcSamples) {
        dataInputWidget->setMethodData(approach, method, useMC, mcSamples);
        updateStatusBar(QString("Method committed: %1 - %2").arg(approach).arg(method));
    });
    
    // 5. Pass velocity model data to DataInput when user explicitly commits selection
    connect(velocityModelWidget, &VelocityModelWidget::modelCommitted,
            this, [this](const QString &modelType) {
        double vp = velocityModelWidget->getHomogeneousVp();
        QVector<VelocityLayer1D> layers = velocityModelWidget->get1DModelData();
        QVector<VelocityPoint3D> points = velocityModelWidget->get3DModelData();

        dataInputWidget->setVelocityData(modelType, vp, layers, points);
        updateStatusBar(QString("Velocity model committed: %1").arg(modelType));
    });
}

void MainWindow::onNewFile() {
    QMessageBox::information(this, "New File", "New file functionality");
}

void MainWindow::onOpenFile() {
    QMessageBox::information(this, "Open File", "Open file functionality");
}

void MainWindow::onSaveFile() {
    QMessageBox::information(this, "Save File", "Save file functionality");
}

void MainWindow::onExit() {
    QApplication::quit();
}

void MainWindow::onAbout() {
    QMessageBox::about(this, "About Earthquake Hypocenter Inversion",
        "<h2>Earthquake Hypocenter Inversion Software</h2>"
        "<p><b>Version:</b> 1.0</p>"
        "<p><b>Developer:</b> Farhan Nuha Zhofiro / 31.22.0006</p>"
        "<p><b>Institution:</b> STMKG</p>"
        "<br>"
        "<p>Software untuk inversi penentuan lokasi hiposenter gempabumi</p>"
        "<p>menggunakan metode:</p>"
        "<ul>"
        "<li>Pendekatan Lokal: Gauss-Newton, Steepest Descent, Levenberg-Marquardt</li>"
        "<li>Pendekatan Global: Grid Search (CPU/GPU), Simulated Annealing, Genetic Algorithm</li>"
        "</ul>"
        "<br>"
        "<p><i>Tugas Akhir - Praktik Inversi Geofisika</i></p>");
}

void MainWindow::updateStatusBar(const QString &message) {
    statusLabel->setText(message);
}
