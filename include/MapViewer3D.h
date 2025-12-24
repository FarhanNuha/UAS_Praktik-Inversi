#ifndef MAPVIEWER3D_H
#define MAPVIEWER3D_H

#include <QWidget>
#include <QVector>
#include "MapViewer2D.h"

class MapViewer3D : public QWidget {
    Q_OBJECT

public:
    explicit MapViewer3D(QWidget *parent = nullptr);
    ~MapViewer3D();

public slots:
    void updateBoundary(const BoundaryData &boundary);
    void updateStations(const QVector<StationData> &stations);

protected:
    void paintEvent(QPaintEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;

private:
    BoundaryData currentBoundary;
    QVector<StationData> stationList;
    
    bool boundarySet;
    bool stationsLoaded;
    
    // Visualization parameters
    double zoomFactor;
    double rotationX;
    double rotationZ;
    bool isRotating;
    QPoint lastMousePos;
    
    void draw3DGrid(QPainter &painter);
    void draw3DStations(QPainter &painter);
    void draw3DAxes(QPainter &painter);
    QPointF project3D(double x, double y, double z, int centerX, int centerY, double scale) const;
    void resetView();
};

#endif // MAPVIEWER3D_H
