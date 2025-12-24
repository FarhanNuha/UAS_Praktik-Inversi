#include "MapViewer3D.h"
#include <QPainter>
#include <QWheelEvent>
#include <QMouseEvent>
#include <cmath>

MapViewer3D::MapViewer3D(QWidget *parent)
    : QWidget(parent), boundarySet(false), stationsLoaded(false),
      zoomFactor(1.0), rotationX(30.0), rotationZ(45.0), isRotating(false)
{
    setMinimumSize(400, 300);
    setMouseTracking(true);
}

MapViewer3D::~MapViewer3D() {
}

void MapViewer3D::updateBoundary(const BoundaryData &boundary) {
    currentBoundary = boundary;
    boundarySet = true;
    update();
}

void MapViewer3D::updateStations(const QVector<StationData> &stations) {
    stationList = stations;
    stationsLoaded = true;
    update();
}

void MapViewer3D::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event);
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Background
    painter.fillRect(rect(), QColor(240, 240, 245));
    
    // Title
    painter.setPen(Qt::black);
    QFont titleFont = painter.font();
    titleFont.setPointSize(12);
    titleFont.setBold(true);
    painter.setFont(titleFont);
    painter.drawText(10, 25, "3D Visualization - Calculating Condition");
    
    if (!boundarySet) {
        painter.drawText(rect(), Qt::AlignCenter, 
            "Set boundary in 'Calculating Condition' tab\nand click 'Commit' to visualize");
        return;
    }
    
    // Draw 3D axes
    draw3DAxes(painter);
    
    // Draw 3D grid and geometry
    draw3DGrid(painter);
    
    // Draw stations if loaded
    if (stationsLoaded) {
        draw3DStations(painter);
    }
    
    // Draw control info overlay
    painter.resetTransform();
    painter.setPen(Qt::black);
    painter.setBrush(QColor(255, 255, 255, 200));
    QRectF infoRect(width() - 200, 10, 190, 70);
    painter.drawRect(infoRect);
    
    painter.drawText(infoRect.adjusted(5, 5, -5, -5), Qt::AlignLeft | Qt::AlignTop,
                    QString("Zoom: %1%\nRotation X: %2°\nRotation Z: %3°")
                        .arg(zoomFactor * 100, 0, 'f', 0)
                        .arg(rotationX, 0, 'f', 0)
                        .arg(rotationZ, 0, 'f', 0));
    
    // Draw controls hint
    painter.setPen(QColor(100, 100, 100));
    painter.drawText(10, height() - 10, "Scroll: Zoom | Drag: Rotate | Double-click: Reset");
}

void MapViewer3D::draw3DGrid(QPainter &painter) {
    int centerX = width() / 2;
    int centerY = height() / 2 + 20;
    double scale = 50 * zoomFactor;
    
    // Calculate grid dimensions
    double xRange = currentBoundary.xMax - currentBoundary.xMin;
    double yRange = currentBoundary.yMax - currentBoundary.yMin;
    double zRange = currentBoundary.zMax - currentBoundary.zMin;
    
    int nX = static_cast<int>(xRange / currentBoundary.dx) + 1;
    int nY = static_cast<int>(yRange / currentBoundary.dy) + 1;
    int nZ = static_cast<int>(zRange / currentBoundary.dz) + 1;
    
    // Draw grid box edges
    painter.setPen(QPen(Qt::darkGray, 2));
    
    // Bottom face
    painter.drawLine(project3D(0, 0, 0, centerX, centerY, scale), 
                    project3D(nX, 0, 0, centerX, centerY, scale));
    painter.drawLine(project3D(nX, 0, 0, centerX, centerY, scale), 
                    project3D(nX, nY, 0, centerX, centerY, scale));
    painter.drawLine(project3D(nX, nY, 0, centerX, centerY, scale), 
                    project3D(0, nY, 0, centerX, centerY, scale));
    painter.drawLine(project3D(0, nY, 0, centerX, centerY, scale), 
                    project3D(0, 0, 0, centerX, centerY, scale));
    
    // Top face
    painter.setPen(QPen(Qt::darkGray, 1));
    painter.drawLine(project3D(0, 0, nZ, centerX, centerY, scale), 
                    project3D(nX, 0, nZ, centerX, centerY, scale));
    painter.drawLine(project3D(nX, 0, nZ, centerX, centerY, scale), 
                    project3D(nX, nY, nZ, centerX, centerY, scale));
    painter.drawLine(project3D(nX, nY, nZ, centerX, centerY, scale), 
                    project3D(0, nY, nZ, centerX, centerY, scale));
    painter.drawLine(project3D(0, nY, nZ, centerX, centerY, scale), 
                    project3D(0, 0, nZ, centerX, centerY, scale));
    
    // Vertical edges
    painter.setPen(QPen(Qt::darkGray, 1.5));
    painter.drawLine(project3D(0, 0, 0, centerX, centerY, scale), 
                    project3D(0, 0, nZ, centerX, centerY, scale));
    painter.drawLine(project3D(nX, 0, 0, centerX, centerY, scale), 
                    project3D(nX, 0, nZ, centerX, centerY, scale));
    painter.drawLine(project3D(nX, nY, 0, centerX, centerY, scale), 
                    project3D(nX, nY, nZ, centerX, centerY, scale));
    painter.drawLine(project3D(0, nY, 0, centerX, centerY, scale), 
                    project3D(0, nY, nZ, centerX, centerY, scale));
    
    // Draw grid lines
    painter.setPen(QPen(QColor(180, 180, 180), 1, Qt::DotLine));
    
    // X-direction grid (vertical planes)
    for (int i = 1; i < nX; ++i) {
        // Bottom
        painter.drawLine(project3D(i, 0, 0, centerX, centerY, scale), 
                        project3D(i, nY, 0, centerX, centerY, scale));
        // Top
        painter.drawLine(project3D(i, 0, nZ, centerX, centerY, scale), 
                        project3D(i, nY, nZ, centerX, centerY, scale));
        // Vertical
        painter.drawLine(project3D(i, 0, 0, centerX, centerY, scale), 
                        project3D(i, 0, nZ, centerX, centerY, scale));
        painter.drawLine(project3D(i, nY, 0, centerX, centerY, scale), 
                        project3D(i, nY, nZ, centerX, centerY, scale));
    }
    
    // Y-direction grid
    for (int j = 1; j < nY; ++j) {
        // Bottom
        painter.drawLine(project3D(0, j, 0, centerX, centerY, scale), 
                        project3D(nX, j, 0, centerX, centerY, scale));
        // Top
        painter.drawLine(project3D(0, j, nZ, centerX, centerY, scale), 
                        project3D(nX, j, nZ, centerX, centerY, scale));
        // Vertical
        painter.drawLine(project3D(0, j, 0, centerX, centerY, scale), 
                        project3D(0, j, nZ, centerX, centerY, scale));
        painter.drawLine(project3D(nX, j, 0, centerX, centerY, scale), 
                        project3D(nX, j, nZ, centerX, centerY, scale));
    }
    
    // Z-direction grid (horizontal planes)
    for (int k = 1; k < nZ; ++k) {
        // X-direction
        painter.drawLine(project3D(0, 0, k, centerX, centerY, scale), 
                        project3D(nX, 0, k, centerX, centerY, scale));
        painter.drawLine(project3D(0, nY, k, centerX, centerY, scale), 
                        project3D(nX, nY, k, centerX, centerY, scale));
        // Y-direction
        painter.drawLine(project3D(0, 0, k, centerX, centerY, scale), 
                        project3D(0, nY, k, centerX, centerY, scale));
        painter.drawLine(project3D(nX, 0, k, centerX, centerY, scale), 
                        project3D(nX, nY, k, centerX, centerY, scale));
    }
    
    // Labels
    painter.setPen(Qt::black);
    QFont labelFont = painter.font();
    labelFont.setPointSizeF(9);
    painter.setFont(labelFont);
    
    QString infoText = QString("Boundary: X[%1→%2] Y[%3→%4] Z[%5→%6] km\nGrid: %7×%8×%9 points (dx=%10, dy=%11, dz=%12 km)")
        .arg(currentBoundary.xMin, 0, 'f', 1).arg(currentBoundary.xMax, 0, 'f', 1)
        .arg(currentBoundary.yMin, 0, 'f', 1).arg(currentBoundary.yMax, 0, 'f', 1)
        .arg(currentBoundary.zMin, 0, 'f', 1).arg(currentBoundary.zMax, 0, 'f', 1)
        .arg(nX).arg(nY).arg(nZ)
        .arg(currentBoundary.dx, 0, 'f', 2)
        .arg(currentBoundary.dy, 0, 'f', 2)
        .arg(currentBoundary.dz, 0, 'f', 2);
    
    painter.fillRect(10, height() - 50, width() - 20, 40, QColor(255, 255, 255, 220));
    painter.drawText(15, height() - 35, width() - 30, 30, Qt::AlignLeft | Qt::TextWordWrap, infoText);
}

void MapViewer3D::draw3DStations(QPainter &painter) {
    int centerX = width() / 2;
    int centerY = height() / 2 + 20;
    double scale = 50 * zoomFactor;
    
    double xRange = currentBoundary.xMax - currentBoundary.xMin;
    double yRange = currentBoundary.yMax - currentBoundary.yMin;
    
    int nX = static_cast<int>(xRange / currentBoundary.dx) + 1;
    int nY = static_cast<int>(yRange / currentBoundary.dy) + 1;
    
    painter.setPen(QPen(Qt::red, 3));
    painter.setBrush(Qt::red);
    
    for (const auto &station : stationList) {
        // Normalize station position to grid coordinates
        double normX = (station.longitude - currentBoundary.xMin) / xRange * nX;
        double normY = (station.latitude - currentBoundary.yMin) / yRange * nY;
        
        QPointF pos = project3D(normX, normY, 0, centerX, centerY, scale);
        painter.drawEllipse(pos, 6, 6);
        
        // Draw vertical line to show position
        QPointF posTop = project3D(normX, normY, 5, centerX, centerY, scale);
        painter.setPen(QPen(Qt::red, 1, Qt::DashLine));
        painter.drawLine(pos, posTop);
        
        painter.setPen(QPen(Qt::red, 3));
        painter.drawEllipse(posTop, 4, 4);
        
        // Draw station name
        painter.setPen(Qt::darkRed);
        QFont font = painter.font();
        font.setPointSizeF(9);
        font.setBold(true);
        painter.setFont(font);
        painter.drawText(pos + QPointF(10, 0), station.name);
        painter.setPen(QPen(Qt::red, 3));
    }
}

void MapViewer3D::draw3DAxes(QPainter &painter) {
    int centerX = width() / 2;
    int centerY = height() / 2 + 20;
    double scale = 50 * zoomFactor;
    
    // Calculate max dimensions for axes
    double xRange = currentBoundary.xMax - currentBoundary.xMin;
    double yRange = currentBoundary.yMax - currentBoundary.yMin;
    double zRange = currentBoundary.zMax - currentBoundary.zMin;
    
    int nX = static_cast<int>(xRange / currentBoundary.dx) + 1;
    int nY = static_cast<int>(yRange / currentBoundary.dy) + 1;
    int nZ = static_cast<int>(zRange / currentBoundary.dz) + 1;
    
    // Origin point
    QPointF origin = project3D(0, 0, 0, centerX, centerY, scale);
    
    // X-axis (red)
    QPointF xEnd = project3D(nX * 1.2, 0, 0, centerX, centerY, scale);
    painter.setPen(QPen(QColor(255, 0, 0), 3));
    painter.drawLine(origin, xEnd);
    painter.setBrush(Qt::red);
    QPolygonF xArrow;
    QPointF xDir = (xEnd - origin);
    double xLen = std::sqrt(xDir.x() * xDir.x() + xDir.y() * xDir.y());
    xDir /= xLen;
    xArrow << xEnd 
           << xEnd - xDir * 15 + QPointF(-xDir.y() * 8, xDir.x() * 8)
           << xEnd - xDir * 15 + QPointF(xDir.y() * 8, -xDir.x() * 8);
    painter.drawPolygon(xArrow);
    
    painter.setPen(Qt::red);
    QFont axisFont = painter.font();
    axisFont.setPointSize(11);
    axisFont.setBold(true);
    painter.setFont(axisFont);
    painter.drawText(xEnd + QPointF(10, 5), QString("X (%1 km)").arg(currentBoundary.xMax - currentBoundary.xMin, 0, 'f', 1));
    
    // Y-axis (green)
    QPointF yEnd = project3D(0, nY * 1.2, 0, centerX, centerY, scale);
    painter.setPen(QPen(QColor(0, 200, 0), 3));
    painter.drawLine(origin, yEnd);
    painter.setBrush(QColor(0, 200, 0));
    QPolygonF yArrow;
    QPointF yDir = (yEnd - origin);
    double yLen = std::sqrt(yDir.x() * yDir.x() + yDir.y() * yDir.y());
    yDir /= yLen;
    yArrow << yEnd 
           << yEnd - yDir * 15 + QPointF(-yDir.y() * 8, yDir.x() * 8)
           << yEnd - yDir * 15 + QPointF(yDir.y() * 8, -yDir.x() * 8);
    painter.drawPolygon(yArrow);
    
    painter.setPen(QColor(0, 150, 0));
    painter.drawText(yEnd + QPointF(10, 5), QString("Y (%1 km)").arg(currentBoundary.yMax - currentBoundary.yMin, 0, 'f', 1));
    
    // Z-axis (blue)
    QPointF zEnd = project3D(0, 0, nZ * 1.2, centerX, centerY, scale);
    painter.setPen(QPen(QColor(0, 0, 255), 3));
    painter.drawLine(origin, zEnd);
    painter.setBrush(Qt::blue);
    QPolygonF zArrow;
    QPointF zDir = (zEnd - origin);
    double zLen = std::sqrt(zDir.x() * zDir.x() + zDir.y() * zDir.y());
    zDir /= zLen;
    zArrow << zEnd 
           << zEnd - zDir * 15 + QPointF(-zDir.y() * 8, zDir.x() * 8)
           << zEnd - zDir * 15 + QPointF(zDir.y() * 8, -zDir.x() * 8);
    painter.drawPolygon(zArrow);
    
    painter.setPen(Qt::blue);
    painter.drawText(zEnd + QPointF(-40, -10), QString("Z (%1 km)").arg(currentBoundary.zMax - currentBoundary.zMin, 0, 'f', 1));
}

QPointF MapViewer3D::project3D(double x, double y, double z, int centerX, int centerY, double scale) const {
    // Rotate around Z-axis (azimuth)
    double radZ = rotationZ * M_PI / 180.0;
    double x1 = x * cos(radZ) - y * sin(radZ);
    double y1 = x * sin(radZ) + y * cos(radZ);
    
    // Rotate around X-axis (elevation)
    double radX = rotationX * M_PI / 180.0;
    double y2 = y1 * cos(radX) - z * sin(radX);
    double z2 = y1 * sin(radX) + z * cos(radX);
    
    // Isometric-style projection
    double isoX = x1 * scale;
    double isoY = y2 * scale * 0.5 - z2 * scale;
    
    return QPointF(centerX + isoX, centerY + isoY);
}

void MapViewer3D::wheelEvent(QWheelEvent *event) {
    double zoomDelta = event->angleDelta().y() > 0 ? 1.1 : 0.9;
    double newZoom = zoomFactor * zoomDelta;
    
    if (newZoom >= 0.3 && newZoom <= 5.0) {
        zoomFactor = newZoom;
        update();
    }
    
    event->accept();
}

void MapViewer3D::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        isRotating = true;
        lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
    }
}

void MapViewer3D::mouseMoveEvent(QMouseEvent *event) {
    if (isRotating) {
        QPoint delta = event->pos() - lastMousePos;
        
        rotationZ += delta.x() * 0.5;
        rotationX += delta.y() * 0.5;
        
        // Clamp rotation X to prevent flipping
        if (rotationX < -89) rotationX = -89;
        if (rotationX > 89) rotationX = 89;
        
        // Normalize rotation Z
        while (rotationZ < 0) rotationZ += 360;
        while (rotationZ >= 360) rotationZ -= 360;
        
        lastMousePos = event->pos();
        update();
    }
}

void MapViewer3D::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        isRotating = false;
        setCursor(Qt::ArrowCursor);
    }
}

void MapViewer3D::mouseDoubleClickEvent(QMouseEvent *event) {
    Q_UNUSED(event);
    resetView();
}

void MapViewer3D::resetView() {
    zoomFactor = 1.0;
    rotationX = 30.0;
    rotationZ = 45.0;
    update();
}
