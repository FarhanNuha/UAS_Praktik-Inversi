#ifndef MAPVIEWER2D_H
#define MAPVIEWER2D_H

#include <QWidget>
#include <QPixmap>
#include <QVector>
#include <QPointF>

struct BoundaryData {
    double xMin, xMax;
    double yMin, yMax;
    double zMin, zMax;
    double dx, dy, dz;
};

struct StationData {
    QString name;
    double latitude;
    double longitude;
    QString arrivalTime;
};

class MapViewer2D : public QWidget {
    Q_OBJECT

public:
    explicit MapViewer2D(QWidget *parent = nullptr);
    ~MapViewer2D();

public slots:
    void updateBoundary(const BoundaryData &boundary);
    void updateStations(const QVector<StationData> &stations);

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;

private:
    QPixmap worldMap;
    QPixmap scaledMap;
    
    BoundaryData currentBoundary;
    QVector<StationData> stationList;
    
    bool boundarySet;
    bool stationsLoaded;
    
    // Zoom and pan
    double zoomFactor;
    QPointF panOffset;
    bool isPanning;
    QPoint lastMousePos;
    
    void loadWorldMap();
    QPointF latLonToPixel(double lat, double lon) const;
    void resetView();
};

#endif // MAPVIEWER2D_H
