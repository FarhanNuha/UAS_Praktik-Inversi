#ifndef VELOCITYMODELWIDGET_H
#define VELOCITYMODELWIDGET_H

#include <QWidget>
#include <QComboBox>
#include <QStackedWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QTextEdit>
#include <QVector>
#include <QLabel>

// Forward declaration
struct BoundaryData;

struct VelocityLayer1D {
    double vp;
    double maxDepth;
};

struct VelocityPoint3D {
    double lat, lon, depth;
    double vp;
};

class Velocity1DPlot : public QWidget {
    Q_OBJECT
    
public:
    explicit Velocity1DPlot(QWidget *parent = nullptr);
    ~Velocity1DPlot();
    void setData(const QVector<VelocityLayer1D> &layers);
    void clearData();
    
protected:
    void paintEvent(QPaintEvent *event) override;
    
private:
    QVector<VelocityLayer1D> velocityLayers;
    bool hasData;
};

class Velocity3DPlot : public QWidget {
    Q_OBJECT
    
public:
    explicit Velocity3DPlot(QWidget *parent = nullptr);
    ~Velocity3DPlot();
    void setData(const QVector<VelocityPoint3D> &points);
    void clearData();
    
protected:
    void paintEvent(QPaintEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    
private:
    QVector<VelocityPoint3D> velocityPoints;
    bool hasData;
    
    double rotationX;
    double rotationZ;
    double zoomFactor;
    bool isRotating;
    QPoint lastMousePos;
    
    QColor getColorForVelocity(double vp, double minVp, double maxVp) const;
    QPointF project3D(double x, double y, double z) const;
};

class VelocityModelWidget : public QWidget {
    Q_OBJECT

public:
    explicit VelocityModelWidget(QWidget *parent = nullptr);
    ~VelocityModelWidget();
    
    QString getModelType() const;
    double getHomogeneousVp() const;
    QString get1DModelPath() const;
    QString get3DModelPath() const;
    QVector<VelocityLayer1D> get1DModelData() const;
    QVector<VelocityPoint3D> get3DModelData() const;
    
    void setBoundary(const BoundaryData &boundary);

private slots:
    void onModelTypeChanged(int index);
    void onLoad1DModel();
    void onLoad3DModel();

private:
    void setupUI();
    
    QComboBox *modelTypeCombo;
    QStackedWidget *modelStack;
    
    // Homogeneous model
    QWidget *homogeneousWidget;
    QLineEdit *vpHomogeneous;
    
    // 1D model
    QWidget *model1DWidget;
    QPushButton *load1DButton;
    QTextEdit *model1DPreview;
    Velocity1DPlot *velocity1DPlot;
    QString model1DFilePath;
    QVector<VelocityLayer1D> model1DData;
    
    // 3D model
    QWidget *model3DWidget;
    QPushButton *load3DButton;
    QTextEdit *model3DPreview;
    Velocity3DPlot *velocity3DPlot;
    QString model3DFilePath;
    QVector<VelocityPoint3D> model3DData;
    QLabel *model3DLocationLabel;
    
    // Boundary for grid validation
    BoundaryData *currentBoundary;
    bool boundarySet;
};

#endif // VELOCITYMODELWIDGET_H
