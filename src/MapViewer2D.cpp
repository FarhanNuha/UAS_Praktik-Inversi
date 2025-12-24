#include "MapViewer2D.h"
#include <QPainter>
#include <QResizeEvent>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QDebug>

MapViewer2D::MapViewer2D(QWidget *parent)
    : QWidget(parent), boundarySet(false), stationsLoaded(false),
      zoomFactor(1.0), panOffset(0, 0), isPanning(false)
{
    setMinimumSize(400, 300);
    setMouseTracking(true);
    loadWorldMap();
}

MapViewer2D::~MapViewer2D() {
}

void MapViewer2D::loadWorldMap() {
    // Try to load from resource first
    if (worldMap.load(":/maps/world.png")) {
        qDebug() << "World map loaded from resources";
    } 
    // Try to load from file system
    else if (worldMap.load("maps/world.png")) {
        qDebug() << "World map loaded from file system";
    } 
    else {
        qDebug() << "Failed to load world map, creating placeholder";
        // Create a simple placeholder
        worldMap = QPixmap(800, 400);
        worldMap.fill(Qt::lightGray);
        
        QPainter p(&worldMap);
        p.setPen(QPen(Qt::darkGray, 2));
        p.drawRect(10, 10, 780, 380);
        p.drawText(worldMap.rect(), Qt::AlignCenter, "World Map\n(Place world.png in maps/ folder)");
    }
    
    scaledMap = worldMap;
}

void MapViewer2D::updateBoundary(const BoundaryData &boundary) {
    currentBoundary = boundary;
    boundarySet = true;
    update();
}

void MapViewer2D::updateStations(const QVector<StationData> &stations) {
    stationList = stations;
    stationsLoaded = true;
    update();
}

void MapViewer2D::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event);
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    
    // Apply zoom and pan transformations
    painter.translate(width() / 2.0, height() / 2.0);
    painter.scale(zoomFactor, zoomFactor);
    painter.translate(-width() / 2.0 + panOffset.x(), -height() / 2.0 + panOffset.y());
    
    // Draw scaled world map
    painter.drawPixmap(0, 0, width(), height(), scaledMap);
    
    // Draw boundary rectangle if set
    if (boundarySet) {
        painter.setPen(QPen(Qt::red, 2 / zoomFactor, Qt::DashLine));
        painter.setBrush(Qt::NoBrush);
        
        QPointF topLeft = latLonToPixel(currentBoundary.yMax, currentBoundary.xMin);
        QPointF bottomRight = latLonToPixel(currentBoundary.yMin, currentBoundary.xMax);
        
        QRectF boundaryRect(topLeft, bottomRight);
        painter.drawRect(boundaryRect);
        
        // Draw grid lines inside boundary
        painter.setPen(QPen(QColor(255, 100, 100, 128), 1 / zoomFactor, Qt::DotLine));
        
        // Calculate number of grid lines
        int nX = static_cast<int>((currentBoundary.xMax - currentBoundary.xMin) / currentBoundary.dx);
        int nY = static_cast<int>((currentBoundary.yMax - currentBoundary.yMin) / currentBoundary.dy);
        
        // Draw vertical grid lines (longitude)
        for (int i = 1; i < nX; ++i) {
            double lon = currentBoundary.xMin + i * currentBoundary.dx;
            QPointF top = latLonToPixel(currentBoundary.yMax, lon);
            QPointF bottom = latLonToPixel(currentBoundary.yMin, lon);
            painter.drawLine(top, bottom);
        }
        
        // Draw horizontal grid lines (latitude)
        for (int j = 1; j < nY; ++j) {
            double lat = currentBoundary.yMin + j * currentBoundary.dy;
            QPointF left = latLonToPixel(lat, currentBoundary.xMin);
            QPointF right = latLonToPixel(lat, currentBoundary.xMax);
            painter.drawLine(left, right);
        }
    }
    
    // Draw stations if loaded
    if (stationsLoaded) {
        painter.setPen(QPen(Qt::blue, 2 / zoomFactor));
        painter.setBrush(Qt::blue);
        
        for (const auto &station : stationList) {
            QPointF pos = latLonToPixel(station.latitude, station.longitude);
            
            // Draw triangle marker
            double size = 8.0 / zoomFactor;
            QPolygonF triangle;
            triangle << pos + QPointF(0, -size)
                    << pos + QPointF(-size * 0.75, size * 0.5)
                    << pos + QPointF(size * 0.75, size * 0.5);
            painter.drawPolygon(triangle);
            
            // Draw station name
            painter.save();
            painter.resetTransform();
            QPointF screenPos = painter.transform().map(pos);
            painter.setPen(Qt::black);
            QFont font = painter.font();
            font.setPointSizeF(9);
            painter.setFont(font);
            painter.drawText(screenPos + QPointF(10, 0), station.name);
            painter.restore();
        }
    }
    
    // Draw info overlay (not affected by zoom/pan)
    painter.resetTransform();
    
    // Draw zoom info
    painter.setPen(Qt::black);
    painter.setBrush(QColor(255, 255, 255, 200));
    QRectF infoRect(10, 10, 150, 50);
    painter.drawRect(infoRect);
    
    painter.drawText(infoRect.adjusted(5, 5, -5, -5), Qt::AlignLeft | Qt::AlignTop,
                    QString("Zoom: %1%\nPan: %2, %3")
                        .arg(zoomFactor * 100, 0, 'f', 0)
                        .arg(panOffset.x(), 0, 'f', 0)
                        .arg(panOffset.y(), 0, 'f', 0));
    
    // Draw grid info if boundary set
    if (boundarySet) {
        painter.setPen(Qt::darkRed);
        painter.setBrush(QColor(255, 220, 220, 220));
        QString gridInfo = QString("Grid: %1×%2 (dx=%3, dy=%4 km)")
            .arg(static_cast<int>((currentBoundary.xMax - currentBoundary.xMin) / currentBoundary.dx) + 1)
            .arg(static_cast<int>((currentBoundary.yMax - currentBoundary.yMin) / currentBoundary.dy) + 1)
            .arg(currentBoundary.dx, 0, 'f', 2)
            .arg(currentBoundary.dy, 0, 'f', 2);
        
        QRectF gridRect(10, height() - 35, 280, 25);
        painter.drawRect(gridRect);
        painter.drawText(gridRect.adjusted(5, 2, -5, -2), Qt::AlignLeft | Qt::AlignVCenter, gridInfo);
    }
    
    // Draw controls hint
    painter.setPen(QColor(100, 100, 100));
    painter.drawText(width() - 220, height() - 10, "Scroll: Zoom | Drag: Pan | Double-click: Reset");
}

void MapViewer2D::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
    scaledMap = worldMap.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

QPointF MapViewer2D::latLonToPixel(double lat, double lon) const {
    // Simple equirectangular projection
    // Longitude: -180 to 180 -> 0 to width
    // Latitude: 90 to -90 -> 0 to height
    
    double x = (lon + 180.0) / 360.0 * width();
    double y = (90.0 - lat) / 180.0 * height();
    
    return QPointF(x, y);
}

void MapViewer2D::wheelEvent(QWheelEvent *event) {
    // Zoom in/out with mouse wheel
    double zoomDelta = event->angleDelta().y() > 0 ? 1.1 : 0.9;
    double newZoom = zoomFactor * zoomDelta;
    
    // Limit zoom range
    if (newZoom >= 0.5 && newZoom <= 10.0) {
        zoomFactor = newZoom;
        update();
    }
    
    event->accept();
}

void MapViewer2D::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        isPanning = true;
        lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
    }
}

void MapViewer2D::mouseMoveEvent(QMouseEvent *event) {
    if (isPanning) {
        QPointF delta = event->pos() - lastMousePos;
        panOffset += delta / zoomFactor;
        lastMousePos = event->pos();
        update();
    }
}

void MapViewer2D::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        isPanning = false;
        setCursor(Qt::ArrowCursor);
    }
}

void MapViewer2D::mouseDoubleClickEvent(QMouseEvent *event) {
    Q_UNUSED(event);
    resetView();
}

void MapViewer2D::resetView() {
    zoomFactor = 1.0;
    panOffset = QPointF(0, 0);
    update();
}
