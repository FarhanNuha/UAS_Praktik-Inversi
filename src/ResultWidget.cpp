#include "ResultWidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QMessageBox>
#include <QDateTime>
#include <QPainter>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QGroupBox>
#include <cmath>

// ============================================================================
// MisfitPlot Implementation
// ============================================================================

MisfitPlot::MisfitPlot(QWidget *parent)
    : QWidget(parent), hasData(false)
{
    setMinimumSize(300, 200);
}

void MisfitPlot::setData(const QVector<double> &iterations, const QVector<double> &misfits) {
    iterData = iterations;
    misfitData = misfits;
    hasData = !iterations.isEmpty() && !misfits.isEmpty();
    update();
    repaint();
}

void MisfitPlot::clearData() {
    iterData.clear();
    misfitData.clear();
    hasData = false;
    update();
}

void MisfitPlot::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event);
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    painter.fillRect(rect(), Qt::white);
    painter.setPen(QPen(Qt::gray, 1));
    painter.drawRect(rect().adjusted(0, 0, -1, -1));
    
    if (!hasData) {
        painter.drawText(rect(), Qt::AlignCenter, "Misfit plot akan muncul setelah komputasi");
        return;
    }
    
    int leftMargin = 60;
    int rightMargin = 20;
    int topMargin = 30;
    int bottomMargin = 50;
    
    int plotWidth = width() - leftMargin - rightMargin;
    int plotHeight = height() - topMargin - bottomMargin;
    
    // Find ranges
    double minIter = iterData.first();
    double maxIter = iterData.last();
    double minMisfit = *std::min_element(misfitData.begin(), misfitData.end());
    double maxMisfit = *std::max_element(misfitData.begin(), misfitData.end());
    
    double misfitRange = maxMisfit - minMisfit;
    minMisfit -= misfitRange * 0.1;
    maxMisfit += misfitRange * 0.1;
    
    // Draw axes
    painter.setPen(QPen(Qt::black, 2));
    painter.drawLine(leftMargin, topMargin, leftMargin, height() - bottomMargin);
    painter.drawLine(leftMargin, height() - bottomMargin, width() - rightMargin, height() - bottomMargin);
    
    // Title
    QFont titleFont = painter.font();
    titleFont.setBold(true);
    titleFont.setPointSize(10);
    painter.setFont(titleFont);
    painter.drawText(QRect(0, 5, width(), 20), Qt::AlignCenter, "Misfit vs Iteration");
    
    // Axis labels
    painter.drawText(QRect(0, height() - 25, width(), 20), Qt::AlignCenter, "Iteration");
    
    painter.save();
    painter.translate(15, height() / 2);
    painter.rotate(-90);
    painter.drawText(QRect(-60, -10, 120, 20), Qt::AlignCenter, "Misfit");
    painter.restore();
    
    QFont labelFont = painter.font();
    labelFont.setPointSize(8);
    painter.setFont(labelFont);
    
    // Draw grid and ticks
    painter.setPen(QPen(QColor(220, 220, 220), 1, Qt::DotLine));
    for (int i = 0; i <= 5; ++i) {
        int y = topMargin + (plotHeight * i / 5);
        painter.drawLine(leftMargin, y, width() - rightMargin, y);
        
        double misfit = maxMisfit - (maxMisfit - minMisfit) * i / 5;
        painter.setPen(Qt::black);
        painter.drawText(QRect(0, y - 10, leftMargin - 5, 20), Qt::AlignRight | Qt::AlignVCenter, 
                        QString::number(misfit, 'e', 2));
        painter.setPen(QPen(QColor(220, 220, 220), 1, Qt::DotLine));
    }
    
    // Draw iteration labels on X-axis
    painter.setPen(Qt::black);
    for (int i = 0; i <= 5 && i < iterData.size(); ++i) {
        int idx = (iterData.size() - 1) * i / 5;
        double x = leftMargin + (iterData[idx] - minIter) / (maxIter - minIter) * plotWidth;
        painter.drawText(QRect(x - 20, height() - bottomMargin + 5, 40, 15), Qt::AlignCenter, 
                        QString::number(static_cast<int>(iterData[idx])));
    }
    
    // Draw line plot
    painter.setPen(QPen(QColor(0, 120, 215), 2));
    for (int i = 0; i < iterData.size() - 1; ++i) {
        double x1 = leftMargin + (iterData[i] - minIter) / (maxIter - minIter) * plotWidth;
        double y1 = height() - bottomMargin - (misfitData[i] - minMisfit) / (maxMisfit - minMisfit) * plotHeight;
        
        double x2 = leftMargin + (iterData[i+1] - minIter) / (maxIter - minIter) * plotWidth;
        double y2 = height() - bottomMargin - (misfitData[i+1] - minMisfit) / (maxMisfit - minMisfit) * plotHeight;
        
        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2));
    }
}

// ============================================================================
// Result2DPlot Implementation
// ============================================================================

Result2DPlot::Result2DPlot(QWidget *parent)
    : QWidget(parent), hasData(false), resultX(0), resultY(0)
{
    setMinimumSize(300, 200);
}

void Result2DPlot::setResult(double x, double y, const QVector<QPointF> &contour, const QVector<StationData> &stations) {
    resultX = x;
    resultY = y;
    contourData = contour;
    stationData = stations;
    hasData = true;
    update();
    repaint();
}

void Result2DPlot::clearData() {
    contourData.clear();
    hasData = false;
    update();
}

void Result2DPlot::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event);
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    painter.fillRect(rect(), QColor(245, 245, 245));
    painter.setPen(QPen(Qt::gray, 1));
    painter.drawRect(rect().adjusted(0, 0, -1, -1));
    
    if (!hasData) {
        painter.drawText(rect(), Qt::AlignCenter, "Hasil 2D dengan kontur misfit\nakan muncul setelah komputasi");
        return;
    }
    
    // Title
    QFont titleFont = painter.font();
    titleFont.setBold(true);
    titleFont.setPointSize(10);
    painter.setFont(titleFont);
    painter.drawText(QRect(0, 5, width(), 20), Qt::AlignCenter, "Result 2D - XY Plane with Misfit Contour");
    
    int leftMargin = 50;
    int rightMargin = 60;
    int topMargin = 30;
    int bottomMargin = 40;
    
    int plotWidth = width() - leftMargin - rightMargin;
    int plotHeight = height() - topMargin - bottomMargin;
    
    // Find bounds from ALL available data: result, stations, and contour
    double minX = resultX, maxX = resultX;
    double minY = resultY, maxY = resultY;
    
    // Include all stations
    for (const auto &sta : stationData) {
        minX = std::min(minX, sta.longitude);
        maxX = std::max(maxX, sta.longitude);
        minY = std::min(minY, sta.latitude);
        maxY = std::max(maxY, sta.latitude);
    }
    
    // Include all contour points
    if (!contourData.isEmpty()) {
        for (const auto &pt : contourData) {
            minX = std::min(minX, pt.x());
            maxX = std::max(maxX, pt.x());
            minY = std::min(minY, pt.y());
            maxY = std::max(maxY, pt.y());
        }
    }
    
    // Add padding (10% on each side)
    double xRange = maxX - minX;
    double yRange = maxY - minY;
    if (xRange < 0.1) xRange = 0.1;
    if (yRange < 0.1) yRange = 0.1;
    
    double padding = 0.1;
    minX -= xRange * padding;
    maxX += xRange * padding;
    minY -= yRange * padding;
    maxY += yRange * padding;
    
    xRange = maxX - minX;
    yRange = maxY - minY;
    
    // Draw axes
    painter.setPen(QPen(Qt::black, 1));
    painter.drawLine(leftMargin, topMargin, leftMargin, topMargin + plotHeight);
    painter.drawLine(leftMargin, topMargin + plotHeight, leftMargin + plotWidth, topMargin + plotHeight);
    
    // Axis labels
    QFont labelFont = painter.font();
    labelFont.setPointSize(8);
    painter.setFont(labelFont);
    painter.drawText(leftMargin + plotWidth / 2 - 30, height() - 15, "Longitude (°)");
    painter.drawText(5, topMargin + plotHeight / 2, "Latitude (°)");
    
    // Draw contour with viridis colormap
    if (!contourData.isEmpty()) {
        // Draw filled contour with gradient colors
        for (int i = 0; i < contourData.size(); ++i) {
            double t = static_cast<double>(i) / contourData.size();
            QColor color = getColorViridis(t);
            
            painter.setPen(QPen(color, 2));
            double x1 = leftMargin + (contourData[i].x() - minX) / xRange * plotWidth;
            double y1 = topMargin + plotHeight - (contourData[i].y() - minY) / yRange * plotHeight;
            
            if (i > 0) {
                double x0 = leftMargin + (contourData[i-1].x() - minX) / xRange * plotWidth;
                double y0 = topMargin + plotHeight - (contourData[i-1].y() - minY) / yRange * plotHeight;
                painter.drawLine(QPointF(x0, y0), QPointF(x1, y1));
            }
        }
        
        // Close contour
        if (contourData.size() > 2) {
            QColor lastColor = getColorViridis(1.0);
            painter.setPen(QPen(lastColor, 2));
            double x1 = leftMargin + (contourData.last().x() - minX) / xRange * plotWidth;
            double y1 = topMargin + plotHeight - (contourData.last().y() - minY) / yRange * plotHeight;
            double x2 = leftMargin + (contourData[0].x() - minX) / xRange * plotWidth;
            double y2 = topMargin + plotHeight - (contourData[0].y() - minY) / yRange * plotHeight;
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2));
        }
    }
    
    // Draw stations
    painter.setPen(QPen(Qt::darkBlue, 1));
    painter.setFont(QFont("Arial", 7));
    for (const auto &sta : stationData) {
        double stationX = leftMargin + (sta.longitude - minX) / xRange * plotWidth;
        double stationY = topMargin + plotHeight - (sta.latitude - minY) / yRange * plotHeight;
        
        // Triangle marker
        painter.setBrush(Qt::darkBlue);
        QPolygonF triangle;
        triangle << QPointF(stationX, stationY - 5)
                 << QPointF(stationX - 4, stationY + 3)
                 << QPointF(stationX + 4, stationY + 3);
        painter.drawPolygon(triangle);
        
        // Station label
        painter.drawText(stationX + 6, stationY - 2, sta.name);
    }
    
    // Draw result point (epicenter)
    double pointX = leftMargin + (resultX - minX) / xRange * plotWidth;
    double pointY = topMargin + plotHeight - (resultY - minY) / yRange * plotHeight;
    painter.setPen(QPen(Qt::red, 2));
    painter.setBrush(Qt::red);
    painter.drawEllipse(QPointF(pointX, pointY), 7, 7);
    
    painter.setPen(Qt::darkRed);
    painter.drawText(pointX + 12, pointY - 8, "Epicenter");
    painter.drawText(pointX + 12, pointY + 4, QString("(%1°, %2°)").arg(resultX, 0, 'f', 2).arg(resultY, 0, 'f', 2));
    
    // Draw colorbar legend
    int barWidth = 15;
    int barHeight = 100;
    int barX = width() - 40;
    int barY = topMargin + 20;
    
    for (int i = 0; i < barHeight; ++i) {
        double t = 1.0 - (double)i / barHeight;
        QColor color = getColorViridis(t);
        painter.setPen(color);
        painter.drawLine(barX, barY + i, barX + barWidth, barY + i);
    }
    
    painter.setPen(Qt::black);
    painter.drawRect(barX, barY, barWidth, barHeight);
    painter.setFont(QFont("Arial", 7));
    painter.drawText(barX - 20, barY - 5, "High");
    painter.drawText(barX - 20, barY + barHeight + 5, "Low");
    painter.drawText(barX - 40, barY + barHeight / 2 - 5, "Misfit");
}

QColor Result2DPlot::getColorViridis(double t) const {
    // Viridis colormap: dark purple -> blue -> green -> yellow
    t = std::max(0.0, std::min(1.0, t));
    
    int r, g, b;
    if (t < 0.25) {
        // Purple to Blue
        double s = t / 0.25;
        r = static_cast<int>(68 * (1 - s) + 33 * s);
        g = static_cast<int>(1 * (1 - s) + 104 * s);
        b = static_cast<int>(84 * (1 - s) + 183 * s);
    } else if (t < 0.5) {
        // Blue to Cyan-Green
        double s = (t - 0.25) / 0.25;
        r = static_cast<int>(33 * (1 - s) + 52 * s);
        g = static_cast<int>(104 * (1 - s) + 151 * s);
        b = static_cast<int>(183 * (1 - s) + 143 * s);
    } else if (t < 0.75) {
        // Cyan-Green to Green-Yellow
        double s = (t - 0.5) / 0.25;
        r = static_cast<int>(52 * (1 - s) + 130 * s);
        g = static_cast<int>(151 * (1 - s) + 200 * s);
        b = static_cast<int>(143 * (1 - s) + 72 * s);
    } else {
        // Green-Yellow to Yellow
        double s = (t - 0.75) / 0.25;
        r = static_cast<int>(130 * (1 - s) + 253 * s);
        g = static_cast<int>(200 * (1 - s) + 231 * s);
        b = static_cast<int>(72 * (1 - s) + 37 * s);
    }
    
    return QColor(r, g, b);
}

// ============================================================================
// Result3DPlot Implementation
// ============================================================================

Result3DPlot::Result3DPlot(QWidget *parent)
    : QWidget(parent), hasData(false), resultX(0), resultY(0), resultZ(0),
      rotationX(30.0), rotationZ(45.0), zoomFactor(1.0), isRotating(false)
{
    setMinimumSize(300, 200);
    setMouseTracking(true);
}

void Result3DPlot::setResult(double x, double y, double z, const QVector<QPointF> &contour, const QVector<StationData> &stations) {
    resultX = x;
    resultY = y;
    resultZ = z;
    contourData = contour;
    stationData = stations;
    hasData = true;
    update();
    repaint();
}

void Result3DPlot::clearData() {
    hasData = false;
    update();
}

void Result3DPlot::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event);
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    painter.fillRect(rect(), QColor(245, 245, 245));
    
    if (!hasData) {
        painter.drawText(rect(), Qt::AlignCenter, "Hasil 3D dengan kontur misfit\nakan muncul setelah komputasi\n\nDrag: Rotate | Scroll: Zoom");
        return;
    }
    
    // Title
    QFont titleFont = painter.font();
    titleFont.setBold(true);
    titleFont.setPointSize(10);
    painter.setFont(titleFont);
    painter.drawText(QRect(0, 5, width(), 20), Qt::AlignCenter, "Result 3D (XYZ) - Red-White-Blue Misfit");
    
    int centerX = width() / 2;
    int centerY = height() / 2 + 10;
    double scale = 80 * zoomFactor;
    
    // Draw grid (reference frame)
    painter.setPen(QPen(QColor(200, 200, 200), 1, Qt::DotLine));
    for (double i = -1.0; i <= 1.0; i += 0.5) {
        for (double j = -1.0; j <= 1.0; j += 0.5) {
            QPointF p1 = project3D(i, j, 0);
            QPointF p2 = project3D(i + 0.5, j, 0);
            painter.drawLine(p1, p2);
        }
    }
    
    // Draw 3D axes
    painter.setPen(QPen(Qt::red, 2));
    painter.drawLine(project3D(0, 0, 0), project3D(1.5, 0, 0));
    QPointF xEnd = project3D(1.5, 0, 0);
    painter.setPen(Qt::red);
    painter.drawText(xEnd + QPointF(5, -5), "X");
    
    painter.setPen(QPen(Qt::green, 2));
    painter.drawLine(project3D(0, 0, 0), project3D(0, 1.5, 0));
    QPointF yEnd = project3D(0, 1.5, 0);
    painter.setPen(Qt::green);
    painter.drawText(yEnd + QPointF(5, -5), "Y");
    
    painter.setPen(QPen(Qt::blue, 2));
    painter.drawLine(project3D(0, 0, 0), project3D(0, 0, 1.5));
    QPointF zEnd = project3D(0, 0, 1.5);
    painter.setPen(Qt::blue);
    painter.drawText(zEnd + QPointF(-15, -5), "Z");
    
    // Draw contour points with red-white-blue colormap
    if (!contourData.isEmpty()) {
        double minMisfit = std::numeric_limits<double>::max();
        double maxMisfit = std::numeric_limits<double>::lowest();
        
        // Find misfit range for normalization
        for (const QPointF &pt : contourData) {
            double misfit = pt.y();
            minMisfit = std::min(minMisfit, misfit);
            maxMisfit = std::max(maxMisfit, misfit);
        }
        
        double misfitRange = maxMisfit - minMisfit;
        if (misfitRange < 1e-10) misfitRange = 1.0;
        
        // Draw each contour point
        for (const QPointF &pt : contourData) {
            double normalizedMisfit = (pt.y() - minMisfit) / misfitRange;
            normalizedMisfit = std::max(0.0, std::min(1.0, normalizedMisfit));
            
            QColor color = getColorRedWhiteBlue(normalizedMisfit);
            painter.setPen(QPen(color, 1));
            painter.setBrush(color);
            
            // Simple 2D projection - use x and misfit as y coordinate
            double normX = (pt.x() - minMisfit) / misfitRange;
            double normY = normalizedMisfit;
            QPointF projPt = project3D(normX, normY, 0.1);
            
            painter.drawEllipse(projPt, 3, 3);
        }
    }
    
    // Draw station markers
    for (const StationData &station : stationData) {
        double normX = station.latitude / 10.0;
        double normY = station.longitude / 10.0;
        double normZ = 0.1; // Fixed elevation for visualization
        
        QPointF stationPos = project3D(normX, normY, normZ);
        
        // Draw triangle for station
        QPolygonF triangle;
        triangle << (stationPos + QPointF(-5, 5))
                 << (stationPos + QPointF(5, 5))
                 << (stationPos + QPointF(0, -5));
        
        painter.setPen(QPen(QColor(100, 150, 200), 1));
        painter.setBrush(QColor(100, 150, 200));
        painter.drawPolygon(triangle);
        
        // Draw station label
        painter.setPen(Qt::darkBlue);
        painter.setFont(QFont("Arial", 7));
        painter.drawText(stationPos + QPointF(8, -5), station.name);
    }
    
    // Draw result point (hypocenter)
    double normX = resultX / 1000.0;
    double normY = resultY / 1000.0;
    double normZ = resultZ / 100.0;
    
    QPointF resultPos = project3D(normX, normY, normZ);
    painter.setPen(QPen(Qt::red, 3));
    painter.setBrush(Qt::red);
    painter.drawEllipse(resultPos, 8, 8);
    
    painter.setPen(Qt::darkRed);
    painter.setFont(QFont("Arial", 8));
    painter.drawText(resultPos + QPointF(12, 0), QString("Hypocenter\n(%1°, %2°, %3 km)")
        .arg(resultX, 0, 'f', 2).arg(resultY, 0, 'f', 2).arg(resultZ, 0, 'f', 1));
    
    // Draw colorbar legend
    int cbX = width() - 30;
    int cbY = 50;
    int cbWidth = 20;
    int cbHeight = 100;
    
    // Colorbar gradient
    for (int i = 0; i < cbHeight; ++i) {
        double t = 1.0 - (double)i / cbHeight; // Top = max, bottom = min
        QColor color = getColorRedWhiteBlue(t);
        painter.fillRect(cbX, cbY + i, cbWidth, 1, color);
    }
    
    // Colorbar border
    painter.setPen(Qt::black);
    painter.drawRect(cbX, cbY, cbWidth, cbHeight);
    painter.drawText(cbX - 40, cbY - 5, "Max Misfit");
    painter.drawText(cbX - 40, cbY + cbHeight + 15, "Min Misfit");
    
    // Draw info
    painter.setPen(Qt::darkGray);
    painter.setFont(QFont("Arial", 7));
    painter.drawText(10, height() - 10, "Drag: Rotate | Scroll: Zoom");
}

QPointF Result3DPlot::project3D(double x, double y, double z) const {
    int centerX = width() / 2;
    int centerY = height() / 2 + 10;
    double scale = 80 * zoomFactor;
    
    double radZ = rotationZ * M_PI / 180.0;
    double x1 = x * cos(radZ) - y * sin(radZ);
    double y1 = x * sin(radZ) + y * cos(radZ);
    
    double radX = rotationX * M_PI / 180.0;
    double y2 = y1 * cos(radX) - z * sin(radX);
    double z2 = y1 * sin(radX) + z * cos(radX);
    
    double isoX = x1 * scale;
    double isoY = y2 * scale * 0.5 - z2 * scale;
    
    return QPointF(centerX + isoX, centerY + isoY);
}

QColor Result3DPlot::getColorRedWhiteBlue(double t) const {
    t = std::max(0.0, std::min(1.0, t));
    
    // Red-White-Blue: Blue (t=0) -> White (t=0.5) -> Red (t=1)
    int r, g, b;
    
    if (t < 0.5) {
        // Blue to White transition
        double s = t / 0.5; // 0 to 1
        r = static_cast<int>(0 + s * 255);
        g = static_cast<int>(0 + s * 255);
        b = 255;
    } else {
        // White to Red transition
        double s = (t - 0.5) / 0.5; // 0 to 1
        r = 255;
        g = static_cast<int>(255 * (1 - s));
        b = static_cast<int>(255 * (1 - s));
    }
    
    return QColor(r, g, b);
}

void Result3DPlot::wheelEvent(QWheelEvent *event) {
    double delta = event->angleDelta().y() > 0 ? 1.1 : 0.9;
    zoomFactor *= delta;
    zoomFactor = std::max(0.3, std::min(3.0, zoomFactor));
    update();
}

void Result3DPlot::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        isRotating = true;
        lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
    }
}

void Result3DPlot::mouseMoveEvent(QMouseEvent *event) {
    if (isRotating) {
        QPoint delta = event->pos() - lastMousePos;
        rotationZ += delta.x() * 0.5;
        rotationX += delta.y() * 0.5;
        
        rotationX = std::max(-89.0, std::min(89.0, rotationX));
        
        while (rotationZ < 0) rotationZ += 360;
        while (rotationZ >= 360) rotationZ -= 360;
        
        lastMousePos = event->pos();
        update();
    }
}

void Result3DPlot::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        isRotating = false;
        setCursor(Qt::ArrowCursor);
    }
}

// ============================================================================
// ResultWidget Implementation
// ============================================================================

ResultWidget::ResultWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

ResultWidget::~ResultWidget() {
}

void ResultWidget::setupUI() {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(5);
    mainLayout->setContentsMargins(5, 5, 5, 5);
    
    // Create 2x2 grid layout
    QGridLayout *gridLayout = new QGridLayout();
    gridLayout->setSpacing(5);
    
    // Grid 1 (Top-Left): Result Text
    QGroupBox *textGroup = new QGroupBox("Hasil Perhitungan", this);
    QVBoxLayout *textLayout = new QVBoxLayout();
    
    resultText = new QTextEdit(this);
    resultText->setReadOnly(true);
    resultText->setPlaceholderText("Hasil perhitungan akan ditampilkan di sini...\n\n"
                                   "Informasi yang akan ditampilkan:\n"
                                   "• Parameter metode yang digunakan\n"
                                   "• Iterasi dan konvergensi\n"
                                   "• Lokasi hasil inversi (X, Y, Z)\n"
                                   "• Origin time\n"
                                   "• Best misfit dan residual\n"
                                   "• Statistik hasil");
    
    QFont font("Courier New", 9);
    resultText->setFont(font);
    textLayout->addWidget(resultText);
    
    textGroup->setLayout(textLayout);
    gridLayout->addWidget(textGroup, 0, 0);
    
    // Grid 2 (Top-Right): Misfit Plot
    QGroupBox *misfitGroup = new QGroupBox("Misfit vs Iteration", this);
    QVBoxLayout *misfitLayout = new QVBoxLayout();
    
    misfitPlot = new MisfitPlot(this);
    misfitLayout->addWidget(misfitPlot);
    
    misfitGroup->setLayout(misfitLayout);
    gridLayout->addWidget(misfitGroup, 0, 1);
    
    // Grid 3 (Bottom-Left): 2D Result
    QGroupBox *result2DGroup = new QGroupBox("Hasil 2D (XY Plane + Kontur Misfit)", this);
    QVBoxLayout *result2DLayout = new QVBoxLayout();
    
    result2DPlot = new Result2DPlot(this);
    result2DLayout->addWidget(result2DPlot);
    
    result2DGroup->setLayout(result2DLayout);
    gridLayout->addWidget(result2DGroup, 1, 0);
    
    // Grid 4 (Bottom-Right): 3D Result
    QGroupBox *result3DGroup = new QGroupBox("Hasil 3D (XYZ + Kontur Misfit)", this);
    QVBoxLayout *result3DLayout = new QVBoxLayout();
    
    result3DPlot = new Result3DPlot(this);
    result3DLayout->addWidget(result3DPlot);
    
    result3DGroup->setLayout(result3DLayout);
    gridLayout->addWidget(result3DGroup, 1, 1);
    
    // Set equal stretch factors
    gridLayout->setRowStretch(0, 1);
    gridLayout->setRowStretch(1, 1);
    gridLayout->setColumnStretch(0, 1);
    gridLayout->setColumnStretch(1, 1);
    
    mainLayout->addLayout(gridLayout);
    
    // Button row
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    
    clearButton = new QPushButton("Clear All", this);
    connect(clearButton, &QPushButton::clicked, this, &ResultWidget::clearResults);
    buttonLayout->addWidget(clearButton);
    
    saveButton = new QPushButton("Save Results", this);
    connect(saveButton, &QPushButton::clicked, this, [this]() {
        QString fileName = QFileDialog::getSaveFileName(
            this,
            "Save Results",
            QString("results_%1.txt").arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss")),
            "Text Files (*.txt);;All Files (*)"
        );
        
        if (fileName.isEmpty()) {
            return;
        }
        
        QFile file(fileName);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QMessageBox::warning(this, "Error", "Cannot save file: " + fileName);
            return;
        }
        
        QTextStream out(&file);
        out << resultText->toPlainText();
        file.close();
        
        QMessageBox::information(this, "Success", "Results saved to: " + fileName);
    });
    buttonLayout->addWidget(saveButton);
    
    buttonLayout->addStretch();
    
    mainLayout->addLayout(buttonLayout);
}

void ResultWidget::appendResult(const QString &text) {
    resultText->append(text);
}

void ResultWidget::clearResults() {
    resultText->clear();
    misfitPlot->clearData();
    result2DPlot->clearData();
    result3DPlot->clearData();
}

void ResultWidget::setMisfitData(const QVector<double> &iterations, const QVector<double> &misfits) {
    misfitPlot->setData(iterations, misfits);
}

void ResultWidget::set2DResult(double x, double y, const QVector<QPointF> &contour, const QVector<StationData> &stations) {
    result2DPlot->setResult(x, y, contour, stations);
}

void ResultWidget::set3DResult(double x, double y, double z, const QVector<QPointF> &contour, const QVector<StationData> &stations) {
    result3DPlot->setResult(x, y, z, contour, stations);
}
