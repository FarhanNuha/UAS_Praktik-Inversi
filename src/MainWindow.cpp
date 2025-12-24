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

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setupUI();
    resize(1280, 720);
    setWindowTitle("Inversi Hiposenter Geofisika");
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
    
    // File Menu
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
    
    // Edit Menu
    QMenu *editMenu = menuBar->addMenu("&Edit");
    QAction *preferencesAction = new QAction("&Preferences", this);
    editMenu->addAction(preferencesAction);
    
    // Settings Menu
    QMenu *settingsMenu = menuBar->addMenu("&Settings");
    QAction *configAction = new QAction("&Configuration", this);
    settingsMenu->addAction(configAction);
    
    // Help Menu
    QMenu *helpMenu = menuBar->addMenu("&Help");
    
    aboutAction = new QAction("&About", this);
    connect(aboutAction, &QAction::triggered, this, &MainWindow::onAbout);
    helpMenu->addAction(aboutAction);
}

void MainWindow::createStatusBar() {
    QStatusBar *statusBar = new QStatusBar(this);
    setStatusBar(statusBar);
    
    // Left side - status message
    statusLabel = new QLabel("Ready", this);
    statusBar->addWidget(statusLabel, 1);
    
    // Right side - author info
    authorLabel = new QLabel("Farhan Nuha Zhofiro / 31.22.0006", this);
    statusBar->addPermanentWidget(authorLabel);
}

void MainWindow::createCentralWidget() {
    // Main horizontal splitter (8:3 ratio)
    mainSplitter = new QSplitter(Qt::Horizontal, this);
    
    // LEFT PANEL (8 parts) - Tab Widget
    leftTabWidget = new QTabWidget(this);
    
    mapViewer2D = new MapViewer2D(this);
    leftTabWidget->addTab(mapViewer2D, "2D View");
    
    mapViewer3D = new MapViewer3D(this);
    leftTabWidget->addTab(mapViewer3D, "3D View");
    
    resultWidget = new ResultWidget(this);
    leftTabWidget->addTab(resultWidget, "Hasil");
    
    mainSplitter->addWidget(leftTabWidget);
    
    // RIGHT PANEL (3 parts) - Vertical split
    rightPanel = new QWidget(this);
    QVBoxLayout *rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->setSpacing(5);
    
    // Top right tab widget
    topRightTabWidget = new QTabWidget(this);
    
    calcConditionWidget = new CalculatingConditionWidget(this);
    topRightTabWidget->addTab(calcConditionWidget, "Calculating Condition");
    
    methodWidget = new MethodWidget(this);
    topRightTabWidget->addTab(methodWidget, "Metode");
    
    rightLayout->addWidget(topRightTabWidget, 1);
    
    // Bottom right tab widget
    bottomRightTabWidget = new QTabWidget(this);
    
    velocityModelWidget = new VelocityModelWidget(this);
    bottomRightTabWidget->addTab(velocityModelWidget, "Velocity Model");
    
    dataInputWidget = new DataInputWidget(this);
    bottomRightTabWidget->addTab(dataInputWidget, "Data Input");
    
    rightLayout->addWidget(bottomRightTabWidget, 1);
    
    mainSplitter->addWidget(rightPanel);
    
    // Set splitter ratio to 8:3
    mainSplitter->setStretchFactor(0, 8);
    mainSplitter->setStretchFactor(1, 3);
    
    setCentralWidget(mainSplitter);
    
    // Connect signals
    connect(calcConditionWidget, &CalculatingConditionWidget::conditionCommitted,
            mapViewer2D, &MapViewer2D::updateBoundary);
    connect(calcConditionWidget, &CalculatingConditionWidget::conditionCommitted,
            mapViewer3D, &MapViewer3D::updateBoundary);
    
    connect(dataInputWidget, &DataInputWidget::stationsLoaded,
            mapViewer2D, &MapViewer2D::updateStations);
    connect(dataInputWidget, &DataInputWidget::stationsLoaded,
            mapViewer3D, &MapViewer3D::updateStations);
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
    QMessageBox::about(this, "About GeoViewer",
        "GeoViewer v1.0\n\n"
        "Seismic Data Processing and Analysis Tool\n\n"
        "Developer: Professional Qt6 C++ Framework\n"
        "Client: Farhan Nuha Zhofiro / 31.22.0006");
}

void MainWindow::updateStatusBar(const QString &message) {
    statusLabel->setText(message);
}
