#ifndef RESULTWIDGET_H
#define RESULTWIDGET_H

#include <QWidget>
#include <QTextEdit>
#include <QPushButton>
#include <QVector>

class MisfitPlot : public QWidget {
    Q_OBJECT
    
public:
    explicit MisfitPlot(QWidget *parent = nullptr);
    void setData(const QVector<double> &iterations, const QVector<double> &misfits);
    void clearData();
    
protected:
    void paintEvent(QPaintEvent *event) override;
    
private:
    QVector<double> iterData;
    QVector<double> misfitData;
    bool hasData;
};

class Result2DPlot : public QWidget {
    Q_OBJECT
    
public:
    explicit Result2DPlot(QWidget *parent = nullptr);
    void setResult(double x, double y, const QVector<QPointF> &contour);
    void clearData();
    
protected:
    void paintEvent(QPaintEvent *event) override;
    
private:
    double resultX, resultY;
    QVector<QPointF> contourData;
    bool hasData;
};

class Result3DPlot : public QWidget {
    Q_OBJECT
    
public:
    explicit Result3DPlot(QWidget *parent = nullptr);
    void setResult(double x, double y, double z);
    void clearData();
    
protected:
    void paintEvent(QPaintEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    
private:
    double resultX, resultY, resultZ;
    bool hasData;
    
    double rotationX;
    double rotationZ;
    double zoomFactor;
    bool isRotating;
    QPoint lastMousePos;
    
    QPointF project3D(double x, double y, double z) const;
};

class ResultWidget : public QWidget {
    Q_OBJECT

public:
    explicit ResultWidget(QWidget *parent = nullptr);
    ~ResultWidget();
    
    void appendResult(const QString &text);
    void clearResults();
    void setMisfitData(const QVector<double> &iterations, const QVector<double> &misfits);
    void set2DResult(double x, double y, const QVector<QPointF> &contour);
    void set3DResult(double x, double y, double z);

private:
    void setupUI();
    
    QTextEdit *resultText;
    MisfitPlot *misfitPlot;
    Result2DPlot *result2DPlot;
    Result3DPlot *result3DPlot;
    
    QPushButton *clearButton;
    QPushButton *saveButton;
};

#endif // RESULTWIDGET_H
