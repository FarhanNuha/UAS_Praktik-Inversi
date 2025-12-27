#include "MapViewer2D.h"
#include <QPainter>
#include <QResizeEvent>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QDebug>
#include <cmath>

// ============================================================================
// MapCanvas Implementation
// ============================================================================

MapCanvas::MapCanvas(QWidget *parent)
    : QWidget(parent), boundarySet(false), stationsLoaded(false),
      zoomFactor(1.0), panOffset(0, 0), isPanning(false)
{
    setMinimumSize(400, 300);
    setMouseTracking(true);
    loadWorldMap();
    zoomToIndonesia(); // Auto zoom to Indonesia on startup
}

MapCanvas::~MapCanvas() {
}

void MapCanvas::loadWorldMap() {
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
        worldMap = QPixmap(800, 400);
        worldMap.fill(Qt::lightGray);
        
        QPainter p(&worldMap);
        p.setPen(QPen(Qt::darkGray, 2));
        p.drawRect(10, 10, 780, 380);
        p.drawText(worldMap.rect(), Qt::AlignCenter, "World Map\n(Place world.png in maps/ folder)");
    }
    
    scaledMap = worldMap;
}

void MapCanvas::updateBoundary(const BoundaryData &boundary) {
    currentBoundary = boundary;
    boundarySet = true;
    update();
}

void MapCanvas::updateStations(const QVector<StationData> &stations) {
    stationList = stations;
    stationsLoaded = true;
    update();
}

void MapCanvas::zoomToIndonesia() {
    // Indonesia bounding box: Lon 95-141, Lat -11 to 6
    double indoLonMin = 95.0, indoLonMax = 141.0;
    double indoLatMin = -11.0, indoLatMax = 6.0;
    
    // Calculate center
    double centerLon = (indoLonMin + indoLonMax) / 2.0; // ~118
    double centerLat = (indoLatMin + indoLatMax) / 2.0; // ~-2.5
    
    // Calculate zoom to fit Indonesia
    double lonRange = indoLonMax - indoLonMin; // 46 degrees
    double latRange = indoLatMax - indoLatMin; // 17 degrees
    
    double worldLonRange = 360.0;
    double worldLatRange = 180.0;
    
    double zoomLon = worldLonRange / lonRange;
    double zoomLat = worldLatRange / latRange;
    
    zoomFactor = std::min(zoomLon, zoomLat) * 0.8; // 0.8 for padding
    
    // Calculate pan offset to center Indonesia
    QPointF centerPixel = latLonToPixel(centerLat, centerLon);
    panOffset = QPointF(width() / 2.0 - centerPixel.x(), height() / 2.0 - centerPixel.y());
    
    update();
}

void MapCanvas::paintEvent(QPaintEvent *event) {
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
    
    // Draw info overlay
    painter.resetTransform();
    
    painter.setPen(Qt::black);
    painter.setBrush(QColor(255, 255, 255, 200));
    QRectF infoRect(10, 10, 150, 50);
    painter.drawRect(infoRect);
    
    painter.drawText(infoRect.adjusted(5, 5, -5, -5), Qt::AlignLeft | Qt::AlignTop,
                    QString("Zoom: %1%\nPan: %2, %3")
                        .arg(zoomFactor * 100, 0, 'f', 0)
                        .arg(panOffset.x(), 0, 'f', 0)
                        .arg(panOffset.y(), 0, 'f', 0));
    
    if (boundarySet) {
        painter.setPen(Qt::darkRed);
        painter.setBrush(QColor(255, 220, 220, 220));
        QString gridInfo = QString("Grid: %1×%2 (spacing=%3 km)")
            .arg(static_cast<int>((currentBoundary.xMax - currentBoundary.xMin) / currentBoundary.gridSpacing) + 1)
            .arg(static_cast<int>((currentBoundary.yMax - currentBoundary.yMin) / currentBoundary.gridSpacing) + 1)
            .arg(currentBoundary.gridSpacing, 0, 'f', 2);
        
        QRectF gridRect(10, height() - 35, 280, 25);
        painter.drawRect(gridRect);
        painter.drawText(gridRect.adjusted(5, 2, -5, -2), Qt::AlignLeft | Qt::AlignVCenter, gridInfo);
    }
    
    painter.setPen(QColor(100, 100, 100));
    painter.drawText(width() - 270, height() - 10, "Scroll: Zoom | Drag: Pan | Double-click: Indonesia");
    
    // Emit view changed signal for scale updates
    QRectF bounds = getVisibleBounds();
    emit viewChanged(bounds.left(), bounds.right(), bounds.bottom(), bounds.top());
}

void MapCanvas::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
    scaledMap = worldMap.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    
    QRectF bounds = getVisibleBounds();
    emit viewChanged(bounds.left(), bounds.right(), bounds.bottom(), bounds.top());
}

QPointF MapCanvas::latLonToPixel(double lat, double lon) const {
    // WGS84 EPSG:4326 projection (equirectangular)
    double x = (lon + 180.0) / 360.0 * width();
    double y = (90.0 - lat) / 180.0 * height();
    return QPointF(x, y);
}

void MapCanvas::pixelToLatLon(const QPointF &pixel, double &lat, double &lon) const {
    lon = (pixel.x() / width()) * 360.0 - 180.0;
    lat = 90.0 - (pixel.y() / height()) * 180.0;
}

QRectF MapCanvas::getVisibleBounds() const {
    // Get corners in screen space
    QPointF topLeft(0, 0);
    QPointF bottomRight(width(), height());
    
    // Transform to map space (inverse of paint transform)
    QTransform transform;
    transform.translate(width() / 2.0, height() / 2.0);
    transform.scale(zoomFactor, zoomFactor);
    transform.translate(-width() / 2.0 + panOffset.x(), -height() / 2.0 + panOffset.y());
    
    QTransform invTransform = transform.inverted();
    topLeft = invTransform.map(topLeft);
    bottomRight = invTransform.map(bottomRight);
    
    double minLat, maxLat, minLon, maxLon;
    pixelToLatLon(topLeft, maxLat, minLon);
    pixelToLatLon(bottomRight, minLat, maxLon);
    
    // Clamp to world bounds
    minLon = std::max(-180.0, std::min(180.0, minLon));
    maxLon = std::max(-180.0, std::min(180.0, maxLon));
    minLat = std::max(-90.0, std::min(90.0, minLat));
    maxLat = std::max(-90.0, std::min(90.0, maxLat));
    
    return QRectF(QPointF(minLon, maxLat), QPointF(maxLon, minLat));
}

void MapCanvas::wheelEvent(QWheelEvent *event) {
    double zoomDelta = event->angleDelta().y() > 0 ? 1.1 : 0.9;
    double newZoom = zoomFactor * zoomDelta;
    
    // Limit zoom range - don't allow zoom out below world view
    if (newZoom >= 1.0 && newZoom <= 20.0) {
        zoomFactor = newZoom;
        update();
    }
    
    event->accept();
}

void MapCanvas::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        isPanning = true;
        lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
    }
}

void MapCanvas::mouseMoveEvent(QMouseEvent *event) {
    if (isPanning) {
        QPointF delta = event->pos() - lastMousePos;
        panOffset += delta / zoomFactor;
        lastMousePos = event->pos();
        update();
    }
}

void MapCanvas::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        isPanning = false;
        setCursor(Qt::ArrowCursor);
    }
}

void MapCanvas::mouseDoubleClickEvent(QMouseEvent *event) {
    Q_UNUSED(event);
    zoomToIndonesia();
}

void MapCanvas::resetView() {
    zoomFactor = 1.0;
    panOffset = QPointF(0, 0);
    update();
}

// ============================================================================
// LatitudeScale Implementation
// ============================================================================

LatitudeScale::LatitudeScale(QWidget *parent)
    : QWidget(parent), minLat(-90), maxLat(90)
{
    setFixedWidth(50);
    setMinimumHeight(100);
}

LatitudeScale::~LatitudeScale() {
}

void LatitudeScale::setRange(double min, double max) {
    minLat = min;
    maxLat = max;
    update();
}

void LatitudeScale::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event);
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Background
    painter.fillRect(rect(), QColor(250, 250, 250));
    painter.setPen(QPen(Qt::gray, 1));
    painter.drawRect(rect().adjusted(0, 0, -1, -1));
    
    // Calculate tick spacing
    double range = maxLat - minLat;
    double tickSpacing = 10.0;
    
    if (range < 5) tickSpacing = 1.0;
    else if (range < 20) tickSpacing = 2.0;
    else if (range < 50) tickSpacing = 5.0;
    else if (range < 100) tickSpacing = 10.0;
    else tickSpacing = 20.0;
    
    // Draw ticks and labels
    int startLat = static_cast<int>(std::floor(minLat / tickSpacing) * tickSpacing);
    int endLat = static_cast<int>(std::ceil(maxLat / tickSpacing) * tickSpacing);
    
    QFont font = painter.font();
    font.setPointSize(8);
    painter.setFont(font);
    
    for (int lat = startLat; lat <= endLat; lat += static_cast<int>(tickSpacing)) {
        if (lat < minLat || lat > maxLat) continue;
        
        double t = (maxLat - lat) / range;
        int y = static_cast<int>(t * height());
        
        // Draw tick
        painter.setPen(QPen(Qt::black, 1));
        painter.drawLine(width() - 8, y, width() - 2, y);
        
        // Draw label
        QString label = QString("%1°").arg(lat);
        QRect textRect(0, y - 10, width() - 10, 20);
        painter.drawText(textRect, Qt::AlignRight | Qt::AlignVCenter, label);
    }
}

// ============================================================================
// LongitudeScale Implementation
// ============================================================================

LongitudeScale::LongitudeScale(QWidget *parent)
    : QWidget(parent), minLon(-180), maxLon(180)
{
    setFixedHeight(30);
    setMinimumWidth(100);
}

LongitudeScale::~LongitudeScale() {
}

void LongitudeScale::setRange(double min, double max) {
    minLon = min;
    maxLon = max;
    update();
}

void LongitudeScale::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event);
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Background
    painter.fillRect(rect(), QColor(250, 250, 250));
    painter.setPen(QPen(Qt::gray, 1));
    painter.drawRect(rect().adjusted(0, 0, -1, -1));
    
    // Calculate tick spacing
    double range = maxLon - minLon;
    double tickSpacing = 10.0;
    
    if (range < 5) tickSpacing = 1.0;
    else if (range < 20) tickSpacing = 2.0;
    else if (range < 50) tickSpacing = 5.0;
    else if (range < 100) tickSpacing = 10.0;
    else tickSpacing = 20.0;
    
    // Draw ticks and labels
    int startLon = static_cast<int>(std::floor(minLon / tickSpacing) * tickSpacing);
    int endLon = static_cast<int>(std::ceil(maxLon / tickSpacing) * tickSpacing);
    
    QFont font = painter.font();
    font.setPointSize(8);
    painter.setFont(font);
    
    for (int lon = startLon; lon <= endLon; lon += static_cast<int>(tickSpacing)) {
        if (lon < minLon || lon > maxLon) continue;
        
        double t = (lon - minLon) / range;
        int x = static_cast<int>(t * width());
        
        // Draw tick
        painter.setPen(QPen(Qt::black, 1));
        painter.drawLine(x, 2, x, 8);
        
        // Draw label
        QString label = QString("%1°").arg(lon);
        QRect textRect(x - 30, 10, 60, 18);
        painter.drawText(textRect, Qt::AlignCenter, label);
    }
}

// ============================================================================
// MapViewer2D Implementation
// ============================================================================

MapViewer2D::MapViewer2D(QWidget *parent)
    : QWidget(parent)
{
    // Create layout with padding
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(2);
    mainLayout->setContentsMargins(5, 5, 5, 5);
    
    QHBoxLayout *mapLayout = new QHBoxLayout();
    mapLayout->setSpacing(2);
    
    // Latitude scale on the left
    latScale = new LatitudeScale(this);
    mapLayout->addWidget(latScale);
    
    // Map canvas in the center
    mapCanvas = new MapCanvas(this);
    mapLayout->addWidget(mapCanvas, 1);
    
    mainLayout->addLayout(mapLayout, 1);
    
    // Longitude scale at the bottom
    QHBoxLayout *lonLayout = new QHBoxLayout();
    lonLayout->setSpacing(2);
    lonLayout->addSpacing(latScale->width() + 2); // Align with map
    
    lonScale = new LongitudeScale(this);
    lonLayout->addWidget(lonScale, 1);
    
    mainLayout->addLayout(lonLayout);
    
    // Connect signals
    connect(mapCanvas, &MapCanvas::viewChanged, this, &MapViewer2D::onViewChanged);
}

MapViewer2D::~MapViewer2D() {
}

void MapViewer2D::updateBoundary(const BoundaryData &boundary) {
    mapCanvas->updateBoundary(boundary);
}

void MapViewer2D::updateStations(const QVector<StationData> &stations) {
    mapCanvas->updateStations(stations);
}

void MapViewer2D::onViewChanged(double minLon, double maxLon, double minLat, double maxLat) {
    latScale->setRange(minLat, maxLat);
    lonScale->setRange(minLon, maxLon);
}
