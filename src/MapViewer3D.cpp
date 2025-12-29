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
    
    // Draw 3D axes with labels
    draw3DAxesWithLabels(painter);
    
    // Draw 3D grid
    draw3DGrid(painter);
    
    // Draw stations if loaded
    if (stationsLoaded) {
        draw3DStations(painter);
    }
    
    // Draw coordinate indicators
    drawCoordinateIndicators(painter);
    
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
    
    // Convert degrees to km (approximate at equator: 1 degree ≈ 111 km)
    double xRangeDeg = currentBoundary.xMax - currentBoundary.xMin;
    double yRangeDeg = currentBoundary.yMax - currentBoundary.yMin;
    
    // Average latitude for more accurate conversion
    double avgLat = (currentBoundary.yMin + currentBoundary.yMax) / 2.0;
    double latRad = avgLat * M_PI / 180.0;
    
    // 1 degree longitude = 111.320 * cos(latitude) km
    // 1 degree latitude = 110.574 km (approximately constant)
    double xRangeKm = xRangeDeg * 111.320 * cos(latRad);
    double yRangeKm = yRangeDeg * 110.574;
    double zRangeKm = currentBoundary.depthMax - currentBoundary.depthMin;
    
    // Calculate grid count based on spacing
    int nX = static_cast<int>(xRangeKm / currentBoundary.gridSpacing) + 1;
    int nY = static_cast<int>(yRangeKm / currentBoundary.gridSpacing) + 1;
    int nZ = static_cast<int>(zRangeKm / currentBoundary.gridSpacing) + 1;
    
    // Calculate scales to make grid proportional
    double maxRange = std::max({xRangeKm, yRangeKm, zRangeKm});
    double baseScale = 300.0 / maxRange * zoomFactor;
    
    double scaleX = baseScale;
    double scaleY = baseScale;
    double scaleZ = baseScale;
    
    // Draw grid box edges
    painter.setPen(QPen(Qt::darkGray, 2));
    
    // Bottom face (depth = 0)
    painter.drawLine(project3D(0, 0, 0, centerX, centerY, scaleX, scaleY, scaleZ), 
                    project3D(nX, 0, 0, centerX, centerY, scaleX, scaleY, scaleZ));
    painter.drawLine(project3D(nX, 0, 0, centerX, centerY, scaleX, scaleY, scaleZ), 
                    project3D(nX, nY, 0, centerX, centerY, scaleX, scaleY, scaleZ));
    painter.drawLine(project3D(nX, nY, 0, centerX, centerY, scaleX, scaleY, scaleZ), 
                    project3D(0, nY, 0, centerX, centerY, scaleX, scaleY, scaleZ));
    painter.drawLine(project3D(0, nY, 0, centerX, centerY, scaleX, scaleY, scaleZ), 
                    project3D(0, 0, 0, centerX, centerY, scaleX, scaleY, scaleZ));
    
    // Top face (max depth)
    painter.setPen(QPen(Qt::darkGray, 1));
    painter.drawLine(project3D(0, 0, nZ, centerX, centerY, scaleX, scaleY, scaleZ), 
                    project3D(nX, 0, nZ, centerX, centerY, scaleX, scaleY, scaleZ));
    painter.drawLine(project3D(nX, 0, nZ, centerX, centerY, scaleX, scaleY, scaleZ), 
                    project3D(nX, nY, nZ, centerX, centerY, scaleX, scaleY, scaleZ));
    painter.drawLine(project3D(nX, nY, nZ, centerX, centerY, scaleX, scaleY, scaleZ), 
                    project3D(0, nY, nZ, centerX, centerY, scaleX, scaleY, scaleZ));
    painter.drawLine(project3D(0, nY, nZ, centerX, centerY, scaleX, scaleY, scaleZ), 
                    project3D(0, 0, nZ, centerX, centerY, scaleX, scaleY, scaleZ));
    
    // Vertical edges
    painter.setPen(QPen(Qt::darkGray, 1.5));
    painter.drawLine(project3D(0, 0, 0, centerX, centerY, scaleX, scaleY, scaleZ), 
                    project3D(0, 0, nZ, centerX, centerY, scaleX, scaleY, scaleZ));
    painter.drawLine(project3D(nX, 0, 0, centerX, centerY, scaleX, scaleY, scaleZ), 
                    project3D(nX, 0, nZ, centerX, centerY, scaleX, scaleY, scaleZ));
    painter.drawLine(project3D(nX, nY, 0, centerX, centerY, scaleX, scaleY, scaleZ), 
                    project3D(nX, nY, nZ, centerX, centerY, scaleX, scaleY, scaleZ));
    painter.drawLine(project3D(0, nY, 0, centerX, centerY, scaleX, scaleY, scaleZ), 
                    project3D(0, nY, nZ, centerX, centerY, scaleX, scaleY, scaleZ));
    
    // Draw grid lines (less dense for clarity)
    painter.setPen(QPen(QColor(180, 180, 180), 1, Qt::DotLine));
    
    // Adaptive grid line density
    int skipX = std::max(1, nX / 10);
    int skipY = std::max(1, nY / 10);
    int skipZ = std::max(1, nZ / 10);
    
    // X-direction grid
    for (int i = skipX; i < nX; i += skipX) {
        painter.drawLine(project3D(i, 0, 0, centerX, centerY, scaleX, scaleY, scaleZ), 
                        project3D(i, nY, 0, centerX, centerY, scaleX, scaleY, scaleZ));
        painter.drawLine(project3D(i, 0, nZ, centerX, centerY, scaleX, scaleY, scaleZ), 
                        project3D(i, nY, nZ, centerX, centerY, scaleX, scaleY, scaleZ));
    }
    
    // Y-direction grid
    for (int j = skipY; j < nY; j += skipY) {
        painter.drawLine(project3D(0, j, 0, centerX, centerY, scaleX, scaleY, scaleZ), 
                        project3D(nX, j, 0, centerX, centerY, scaleX, scaleY, scaleZ));
        painter.drawLine(project3D(0, j, nZ, centerX, centerY, scaleX, scaleY, scaleZ), 
                        project3D(nX, j, nZ, centerX, centerY, scaleX, scaleY, scaleZ));
    }
    
    // Z-direction grid (horizontal planes)
    for (int k = skipZ; k < nZ; k += skipZ) {
        painter.drawLine(project3D(0, 0, k, centerX, centerY, scaleX, scaleY, scaleZ), 
                        project3D(nX, 0, k, centerX, centerY, scaleX, scaleY, scaleZ));
        painter.drawLine(project3D(0, nY, k, centerX, centerY, scaleX, scaleY, scaleZ), 
                        project3D(nX, nY, k, centerX, centerY, scaleX, scaleY, scaleZ));
        painter.drawLine(project3D(0, 0, k, centerX, centerY, scaleX, scaleY, scaleZ), 
                        project3D(0, nY, k, centerX, centerY, scaleX, scaleY, scaleZ));
        painter.drawLine(project3D(nX, 0, k, centerX, centerY, scaleX, scaleY, scaleZ), 
                        project3D(nX, nY, k, centerX, centerY, scaleX, scaleY, scaleZ));
    }
}

void MapViewer3D::draw3DStations(QPainter &painter) {
    int centerX = width() / 2;
    int centerY = height() / 2 + 20;
    
    // Convert degrees to km
    double avgLat = (currentBoundary.yMin + currentBoundary.yMax) / 2.0;
    double latRad = avgLat * M_PI / 180.0;
    double xRangeDeg = currentBoundary.xMax - currentBoundary.xMin;
    double yRangeDeg = currentBoundary.yMax - currentBoundary.yMin;
    double xRangeKm = xRangeDeg * 111.320 * cos(latRad);
    double yRangeKm = yRangeDeg * 110.574;
    double zRangeKm = currentBoundary.depthMax - currentBoundary.depthMin;
    
    int nX = static_cast<int>(xRangeKm / currentBoundary.gridSpacing) + 1;
    int nY = static_cast<int>(yRangeKm / currentBoundary.gridSpacing) + 1;
    int nZ = static_cast<int>(zRangeKm / currentBoundary.gridSpacing) + 1;
    
    double maxRange = std::max({xRangeKm, yRangeKm, zRangeKm});
    double baseScale = 300.0 / maxRange * zoomFactor;
    
    double scaleX = baseScale;
    double scaleY = baseScale;
    double scaleZ = baseScale;
    
    painter.setPen(QPen(Qt::red, 3));
    painter.setBrush(Qt::red);
    
    for (const auto &station : stationList) {
        // Convert station lat/lon to km offset from boundary min
        double stationLonKm = (station.longitude - currentBoundary.xMin) * 111.320 * cos(latRad);
        double stationLatKm = (station.latitude - currentBoundary.yMin) * 110.574;
        
        // Normalize to grid indices
        double normX = stationLonKm / currentBoundary.gridSpacing;
        double normY = stationLatKm / currentBoundary.gridSpacing;
        
        QPointF pos = project3D(normX, normY, 0, centerX, centerY, scaleX, scaleY, scaleZ);
        painter.drawEllipse(pos, 6, 6);
        
        // Draw vertical line
        QPointF posDepth = project3D(normX, normY, nZ * 0.1, centerX, centerY, scaleX, scaleY, scaleZ);
        painter.setPen(QPen(Qt::red, 1, Qt::DashLine));
        painter.drawLine(pos, posDepth);
        
        painter.setPen(QPen(Qt::red, 3));
        painter.drawEllipse(posDepth, 4, 4);
        
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

void MapViewer3D::draw3DAxesWithLabels(QPainter &painter) {
    int centerX = width() / 2;
    int centerY = height() / 2 + 20;
    
    double xRange = currentBoundary.xMax - currentBoundary.xMin;
    double yRange = currentBoundary.yMax - currentBoundary.yMin;
    double zRange = currentBoundary.depthMax - currentBoundary.depthMin;
    
    int nX = static_cast<int>(xRange / currentBoundary.gridSpacing) + 1;
    int nY = static_cast<int>(yRange / currentBoundary.gridSpacing) + 1;
    int nZ = static_cast<int>(zRange / currentBoundary.gridSpacing) + 1;
    
    double maxRange = std::max({xRange, yRange, zRange});
    double baseScale = 300.0 / maxRange * zoomFactor;
    
    double scaleX = baseScale * (xRange / maxRange);
    double scaleY = baseScale * (yRange / maxRange);
    double scaleZ = baseScale * (zRange / maxRange);
    
    // Origin point
    QPointF origin = project3D(0, 0, 0, centerX, centerY, scaleX, scaleY, scaleZ);
    
    // X-axis (red) - Longitude
    QPointF xEnd = project3D(nX * 1.3, 0, 0, centerX, centerY, scaleX, scaleY, scaleZ);
    painter.setPen(QPen(QColor(255, 0, 0), 3));
    painter.drawLine(origin, xEnd);
    painter.setBrush(Qt::red);
    
    QPointF xDir = (xEnd - origin);
    double xLen = std::sqrt(xDir.x() * xDir.x() + xDir.y() * xDir.y());
    xDir /= xLen;
    QPolygonF xArrow;
    xArrow << xEnd 
           << xEnd - xDir * 15 + QPointF(-xDir.y() * 8, xDir.x() * 8)
           << xEnd - xDir * 15 + QPointF(xDir.y() * 8, -xDir.x() * 8);
    painter.drawPolygon(xArrow);
    
    painter.setPen(Qt::red);
    QFont axisFont = painter.font();
    axisFont.setPointSize(10);
    axisFont.setBold(true);
    painter.setFont(axisFont);
    painter.drawText(xEnd + QPointF(10, 5), QString("X (Lon) %1 km").arg(xRange, 0, 'f', 1));
    
    // Y-axis (green) - Latitude
    QPointF yEnd = project3D(0, nY * 1.3, 0, centerX, centerY, scaleX, scaleY, scaleZ);
    painter.setPen(QPen(QColor(0, 200, 0), 3));
    painter.drawLine(origin, yEnd);
    painter.setBrush(QColor(0, 200, 0));
    
    QPointF yDir = (yEnd - origin);
    double yLen = std::sqrt(yDir.x() * yDir.x() + yDir.y() * yDir.y());
    yDir /= yLen;
    QPolygonF yArrow;
    yArrow << yEnd 
           << yEnd - yDir * 15 + QPointF(-yDir.y() * 8, yDir.x() * 8)
           << yEnd - yDir * 15 + QPointF(yDir.y() * 8, -yDir.x() * 8);
    painter.drawPolygon(yArrow);
    
    painter.setPen(QColor(0, 150, 0));
    painter.drawText(yEnd + QPointF(10, 5), QString("Y (Lat) %1 km").arg(yRange, 0, 'f', 1));
    
    // Z-axis (blue) - Depth
    QPointF zEnd = project3D(0, 0, nZ * 1.3, centerX, centerY, scaleX, scaleY, scaleZ);
    painter.setPen(QPen(QColor(0, 0, 255), 3));
    painter.drawLine(origin, zEnd);
    painter.setBrush(Qt::blue);
    
    QPointF zDir = (zEnd - origin);
    double zLen = std::sqrt(zDir.x() * zDir.x() + zDir.y() * zDir.y());
    zDir /= zLen;
    QPolygonF zArrow;
    zArrow << zEnd 
           << zEnd - zDir * 15 + QPointF(-zDir.y() * 8, zDir.x() * 8)
           << zEnd - zDir * 15 + QPointF(zDir.y() * 8, -zDir.x() * 8);
    painter.drawPolygon(zArrow);
    
    painter.setPen(Qt::blue);
    painter.drawText(zEnd + QPointF(-50, -10), QString("Z (Depth) %1 km").arg(zRange, 0, 'f', 1));
}

void MapViewer3D::drawCoordinateIndicators(QPainter &painter) {
    painter.resetTransform();
    
    painter.setPen(Qt::black);
    QFont labelFont = painter.font();
    labelFont.setPointSizeF(9);
    painter.setFont(labelFont);
    
    // Convert degrees to km
    double avgLat = (currentBoundary.yMin + currentBoundary.yMax) / 2.0;
    double latRad = avgLat * M_PI / 180.0;
    
    double xRangeDeg = currentBoundary.xMax - currentBoundary.xMin;
    double yRangeDeg = currentBoundary.yMax - currentBoundary.yMin;
    double xRangeKm = xRangeDeg * 111.320 * cos(latRad);
    double yRangeKm = yRangeDeg * 110.574;
    double zRangeKm = currentBoundary.depthMax - currentBoundary.depthMin;
    
    int nX = static_cast<int>(xRangeKm / currentBoundary.gridSpacing) + 1;
    int nY = static_cast<int>(yRangeKm / currentBoundary.gridSpacing) + 1;
    int nZ = static_cast<int>(zRangeKm / currentBoundary.gridSpacing) + 1;
    
    QString infoText = QString(
        "Coordinate Range:\n"
        "Lon: [%1°, %2°] → %3 km\n"
        "Lat: [%4°, %5°] → %6 km\n"
        "Depth: [%7, %8] km\n"
        "Grid: %9×%10×%11 (%12 km spacing)")
        .arg(currentBoundary.xMin, 0, 'f', 2).arg(currentBoundary.xMax, 0, 'f', 2).arg(xRangeKm, 0, 'f', 1)
        .arg(currentBoundary.yMin, 0, 'f', 2).arg(currentBoundary.yMax, 0, 'f', 2).arg(yRangeKm, 0, 'f', 1)
        .arg(currentBoundary.depthMin, 0, 'f', 1).arg(currentBoundary.depthMax, 0, 'f', 1)
        .arg(nX).arg(nY).arg(nZ)
        .arg(currentBoundary.gridSpacing, 0, 'f', 2);
    
    painter.fillRect(10, 40, 220, 130, QColor(255, 255, 255, 230));
    painter.drawRect(10, 40, 220, 130);
    painter.drawText(15, 55, 210, 115, Qt::AlignLeft | Qt::TextWordWrap, infoText);
}

QPointF MapViewer3D::project3D(double x, double y, double z, int centerX, int centerY, 
                                double scaleX, double scaleY, double scaleZ) const {
    // Rotate around Z-axis (azimuth)
    double radZ = rotationZ * M_PI / 180.0;
    double x1 = x * cos(radZ) - y * sin(radZ);
    double y1 = x * sin(radZ) + y * cos(radZ);
    
    // Rotate around X-axis (elevation)
    double radX = rotationX * M_PI / 180.0;
    double y2 = y1 * cos(radX) - z * sin(radX);
    double z2 = y1 * sin(radX) + z * cos(radX);
    
    // Isometric-style projection
    // NOTE: z is positive downward (depth), so we ADD z2 instead of subtract
    double isoX = x1 * scaleX;
    double isoY = y2 * scaleY * 0.5 + z2 * scaleZ;  // Changed from minus to plus
    
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
        
        // Clamp rotation X
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
