#ifndef MAPVIEWER2D_H
#define MAPVIEWER2D_H

#include <QWidget>
#include <QPixmap>
#include <QVector>
#include <QPointF>
#include <QVBoxLayout>
#include <QHBoxLayout>

struct BoundaryData {
    double xMin, xMax;
    double yMin, yMax;
    double depthMin, depthMax;
    double gridSpacing;
};

struct StationData {
    QString name;
    double latitude;
    double longitude;
    QString arrivalTime;
};

class MapCanvas : public QWidget {
    Q_OBJECT
    
public:
    explicit MapCanvas(QWidget *parent = nullptr);
    ~MapCanvas();
    void updateBoundary(const BoundaryData &boundary);
    void updateStations(const QVector<StationData> &stations);
    
signals:
    void viewChanged(double minLon, double maxLon, double minLat, double maxLat);
    
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
    void pixelToLatLon(const QPointF &pixel, double &lat, double &lon) const;
    void resetView();
    void zoomToIndonesia();
    QRectF getVisibleBounds() const;
};

class LatitudeScale : public QWidget {
    Q_OBJECT
    
public:
    explicit LatitudeScale(QWidget *parent = nullptr);
    ~LatitudeScale();
    void setRange(double minLat, double maxLat);
    
protected:
    void paintEvent(QPaintEvent *event) override;
    
private:
    double minLat, maxLat;
};

class LongitudeScale : public QWidget {
    Q_OBJECT
    
public:
    explicit LongitudeScale(QWidget *parent = nullptr);
    ~LongitudeScale();
    void setRange(double minLon, double maxLon);
    
protected:
    void paintEvent(QPaintEvent *event) override;
    
private:
    double minLon, maxLon;
};

class MapViewer2D : public QWidget {
    Q_OBJECT

public:
    explicit MapViewer2D(QWidget *parent = nullptr);
    ~MapViewer2D();

public slots:
    void updateBoundary(const BoundaryData &boundary);
    void updateStations(const QVector<StationData> &stations);

private slots:
    void onViewChanged(double minLon, double maxLon, double minLat, double maxLat);

private:
    MapCanvas *mapCanvas;
    LatitudeScale *latScale;
    LongitudeScale *lonScale;
};

#endif // MAPVIEWER2D_H
