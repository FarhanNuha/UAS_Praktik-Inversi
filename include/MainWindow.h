#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMenuBar>
#include <QStatusBar>
#include <QSplitter>
#include <QTabWidget>
#include <QLabel>

class MapViewer2D;
class MapViewer3D;
class CalculatingConditionWidget;
class MethodWidget;
class VelocityModelWidget;
class DataInputWidget;
class ResultWidget;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onNewFile();
    void onOpenFile();
    void onSaveFile();
    void onExit();
    void onAbout();
    void updateStatusBar(const QString &message);

private:
    void setupUI();
    void createMenuBar();
    void createStatusBar();
    void createCentralWidget();

    // Menu actions
    QAction *newAction;
    QAction *openAction;
    QAction *saveAction;
    QAction *exitAction;
    QAction *aboutAction;

    // Status bar
    QLabel *statusLabel;
    QLabel *authorLabel;

    // Main layout components
    QSplitter *mainSplitter;
    
    // Left panel (8 parts)
    QTabWidget *leftTabWidget;
    MapViewer2D *mapViewer2D;
    MapViewer3D *mapViewer3D;
    ResultWidget *resultWidget;
    
    // Right panel (3 parts)
    QWidget *rightPanel;
    QTabWidget *topRightTabWidget;
    QTabWidget *bottomRightTabWidget;
    
    // Right panel widgets
    CalculatingConditionWidget *calcConditionWidget;
    MethodWidget *methodWidget;
    VelocityModelWidget *velocityModelWidget;
    DataInputWidget *dataInputWidget;
};

#endif // MAINWINDOW_H
