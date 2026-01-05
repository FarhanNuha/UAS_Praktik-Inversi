#include "VelocityModelWidget.h"
#include "MapViewer2D.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QFileDialog>
#include <QMessageBox>
#include <QFile>
#include <QTextStream>
#include <QDoubleValidator>
#include <QPainter>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QSplitter>
#include <cmath>

// ============================================================================
// Velocity1DPlot Implementation
// ============================================================================

Velocity1DPlot::Velocity1DPlot(QWidget *parent)
    : QWidget(parent), hasData(false)
{
    setMinimumSize(300, 400);
}

Velocity1DPlot::~Velocity1DPlot() {
}

void Velocity1DPlot::setData(const QVector<VelocityLayer1D> &layers) {
    velocityLayers = layers;
    hasData = !layers.isEmpty();
    update();
}

void Velocity1DPlot::clearData() {
    velocityLayers.clear();
    hasData = false;
    update();
}

void Velocity1DPlot::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event);
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Background
    painter.fillRect(rect(), Qt::white);
    painter.setPen(QPen(Qt::gray, 1));
    painter.drawRect(rect().adjusted(0, 0, -1, -1));
    
    if (!hasData) {
        painter.drawText(rect(), Qt::AlignCenter, "Load 1D model to display plot");
        return;
    }
    
    // Margins
    int leftMargin = 60;
    int rightMargin = 20;
    int topMargin = 30;
    int bottomMargin = 50;
    
    int plotWidth = width() - leftMargin - rightMargin;
    int plotHeight = height() - topMargin - bottomMargin;
    
    // Find ranges
    double minVp = 1e9, maxVp = -1e9;
    double maxDepth = 0;
    
    for (const auto &layer : velocityLayers) {
        minVp = std::min(minVp, layer.vp);
        maxVp = std::max(maxVp, layer.vp);
        maxDepth = std::max(maxDepth, layer.maxDepth);
    }
    
    // Add padding to ranges
    double vpRange = maxVp - minVp;
    minVp -= vpRange * 0.1;
    maxVp += vpRange * 0.1;
    
    // Draw axes
    painter.setPen(QPen(Qt::black, 2));
    painter.drawLine(leftMargin, topMargin, leftMargin, height() - bottomMargin);
    painter.drawLine(leftMargin, height() - bottomMargin, width() - rightMargin, height() - bottomMargin);
    
    // Draw title
    QFont titleFont = painter.font();
    titleFont.setBold(true);
    titleFont.setPointSize(11);
    painter.setFont(titleFont);
    painter.drawText(QRect(0, 5, width(), 20), Qt::AlignCenter, "1D Velocity Model");
    
    // Draw Y-axis label (Depth)
    painter.save();
    painter.translate(15, height() / 2);
    painter.rotate(-90);
    painter.drawText(QRect(-80, -10, 160, 20), Qt::AlignCenter, "Depth (km)");
    painter.restore();
    
    // Draw X-axis label (Vp)
    painter.drawText(QRect(0, height() - 25, width(), 20), Qt::AlignCenter, "Vp (km/s)");
    
    QFont labelFont = painter.font();
    labelFont.setPointSize(9);
    painter.setFont(labelFont);
    
    // Draw Y-axis ticks and labels (Depth)
    int numYTicks = 5;
    for (int i = 0; i <= numYTicks; ++i) {
        double depth = (maxDepth / numYTicks) * i;
        int y = topMargin + (plotHeight * i / numYTicks);
        
        painter.setPen(QPen(Qt::black, 1));
        painter.drawLine(leftMargin - 5, y, leftMargin, y);
        
        QString label = QString::number(depth, 'f', 1);
        painter.drawText(QRect(0, y - 10, leftMargin - 10, 20), Qt::AlignRight | Qt::AlignVCenter, label);
        
        // Grid line
        painter.setPen(QPen(QColor(220, 220, 220), 1, Qt::DotLine));
        painter.drawLine(leftMargin, y, width() - rightMargin, y);
    }
    
    // Draw X-axis ticks and labels (Vp)
    int numXTicks = 5;
    for (int i = 0; i <= numXTicks; ++i) {
        double vp = minVp + (maxVp - minVp) * i / numXTicks;
        int x = leftMargin + (plotWidth * i / numXTicks);
        
        painter.setPen(QPen(Qt::black, 1));
        painter.drawLine(x, height() - bottomMargin, x, height() - bottomMargin + 5);
        
        QString label = QString::number(vp, 'f', 1);
        painter.drawText(QRect(x - 30, height() - bottomMargin + 10, 60, 20), Qt::AlignCenter, label);
        
        // Grid line
        painter.setPen(QPen(QColor(220, 220, 220), 1, Qt::DotLine));
        painter.drawLine(x, topMargin, x, height() - bottomMargin);
    }
    
    // Draw stairs plot
    painter.setPen(QPen(QColor(0, 100, 200), 3));
    
    double prevDepth = 0;
    for (int i = 0; i < velocityLayers.size(); ++i) {
        double vp = velocityLayers[i].vp;
        double depth = velocityLayers[i].maxDepth;
        
        // Calculate pixel positions
        int x1 = leftMargin + static_cast<int>((vp - minVp) / (maxVp - minVp) * plotWidth);
        int y1 = topMargin + static_cast<int>((prevDepth / maxDepth) * plotHeight);
        int y2 = topMargin + static_cast<int>((depth / maxDepth) * plotHeight);
        
        // Horizontal line (constant velocity)
        painter.drawLine(x1, y1, x1, y2);
        
        // Vertical line to next layer
        if (i < velocityLayers.size() - 1) {
            double nextVp = velocityLayers[i + 1].vp;
            int x2 = leftMargin + static_cast<int>((nextVp - minVp) / (maxVp - minVp) * plotWidth);
            painter.drawLine(x1, y2, x2, y2);
        }
        
        prevDepth = depth;
    }
}

// ============================================================================
// Velocity3DPlot Implementation
// ============================================================================

Velocity3DPlot::Velocity3DPlot(QWidget *parent)
    : QWidget(parent), hasData(false), rotationX(30.0), rotationZ(45.0),
      zoomFactor(1.0), isRotating(false)
{
    setMinimumSize(400, 400);
    setMouseTracking(true);
}

Velocity3DPlot::~Velocity3DPlot() {
}

void Velocity3DPlot::setData(const QVector<VelocityPoint3D> &points) {
    velocityPoints = points;
    hasData = !points.isEmpty();
    update();
}

void Velocity3DPlot::clearData() {
    velocityPoints.clear();
    hasData = false;
    update();
}

// Bagian paintEvent di Velocity3DPlot yang diperbaiki
void Velocity3DPlot::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event);
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Background
    painter.fillRect(rect(), QColor(245, 245, 245));
    
    if (!hasData) {
        painter.drawText(rect(), Qt::AlignCenter, "Load 3D model to display visualization");
        return;
    }
    
    // Find ranges
    double minLat = 1e9, maxLat = -1e9;
    double minLon = 1e9, maxLon = -1e9;
    double minDepth = 1e9, maxDepth = -1e9;
    double minVp = 1e9, maxVp = -1e9;
    
    for (const auto &pt : velocityPoints) {
        minLat = std::min(minLat, pt.lat);
        maxLat = std::max(maxLat, pt.lat);
        minLon = std::min(minLon, pt.lon);
        maxLon = std::max(maxLon, pt.lon);
        minDepth = std::min(minDepth, pt.depth);
        maxDepth = std::max(maxDepth, pt.depth);
        minVp = std::min(minVp, pt.vp);
        maxVp = std::max(maxVp, pt.vp);
    }
    
    // Draw title
    painter.setPen(Qt::black);
    QFont titleFont = painter.font();
    titleFont.setBold(true);
    titleFont.setPointSize(11);
    painter.setFont(titleFont);
    painter.drawText(QRect(0, 5, width(), 20), Qt::AlignCenter, "3D Velocity Model");
    
    // Draw points with colors
    for (const auto &pt : velocityPoints) {
        // Normalize coordinates to [0, 1] - FIXED: use lat/lon/depth
        double nx = (pt.lon - minLon) / (maxLon - minLon);
        double ny = (pt.lat - minLat) / (maxLat - minLat);
        double nz = (pt.depth - minDepth) / (maxDepth - minDepth);
        
        QPointF screenPos = project3D(nx, ny, nz);
        
        QColor color = getColorForVelocity(pt.vp, minVp, maxVp);
        painter.setBrush(color);
        painter.setPen(Qt::NoPen);
        painter.drawEllipse(screenPos, 3, 3);
    }
    
    // Draw colorbar
    int barWidth = 30;
    int barHeight = 200;
    int barX = width() - 60;
    int barY = height() / 2 - barHeight / 2;
    
    for (int i = 0; i < barHeight; ++i) {
        double t = 1.0 - (double)i / barHeight;
        double vp = minVp + t * (maxVp - minVp);
        QColor color = getColorForVelocity(vp, minVp, maxVp);
        
        painter.setPen(color);
        painter.drawLine(barX, barY + i, barX + barWidth, barY + i);
    }
    
    painter.setPen(Qt::black);
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(barX, barY, barWidth, barHeight);
    
    // Colorbar labels
    QFont labelFont = painter.font();
    labelFont.setPointSize(8);
    painter.setFont(labelFont);
    
    painter.drawText(QRect(barX - 50, barY - 15, 80, 15), Qt::AlignCenter, QString::number(maxVp, 'f', 1));
    painter.drawText(QRect(barX - 50, barY + barHeight, 80, 15), Qt::AlignCenter, QString::number(minVp, 'f', 1));
    painter.drawText(QRect(barX - 50, barY + barHeight + 15, 80, 15), Qt::AlignCenter, "Vp (km/s)");
    
    // Draw controls hint
    painter.setPen(QColor(100, 100, 100));
    painter.drawText(10, height() - 10, "Drag: Rotate | Scroll: Zoom");
}

QColor Velocity3DPlot::getColorForVelocity(double vp, double minVp, double maxVp) const {
    // Red-White-Blue colormap
    double t = (vp - minVp) / (maxVp - minVp);
    
    int r, g, b;
    if (t < 0.5) {
        // Blue to White
        double s = t * 2.0;
        r = static_cast<int>(0 + s * 255);
        g = static_cast<int>(0 + s * 255);
        b = 255;
    } else {
        // White to Red
        double s = (t - 0.5) * 2.0;
        r = 255;
        g = static_cast<int>(255 - s * 255);
        b = static_cast<int>(255 - s * 255);
    }
    
    return QColor(r, g, b);
}

QPointF Velocity3DPlot::project3D(double x, double y, double z) const {
    int centerX = width() / 2;
    int centerY = height() / 2;
    double scale = 300.0 * zoomFactor;
    
    // Rotate around Z-axis
    double radZ = rotationZ * M_PI / 180.0;
    double x1 = x * cos(radZ) - y * sin(radZ);
    double y1 = x * sin(radZ) + y * cos(radZ);
    
    // Rotate around X-axis
    double radX = rotationX * M_PI / 180.0;
    double y2 = y1 * cos(radX) - z * sin(radX);
    double z2 = y1 * sin(radX) + z * cos(radX);
    
    double isoX = x1 * scale;
    double isoY = y2 * scale * 0.5 - z2 * scale;
    
    return QPointF(centerX + isoX, centerY + isoY);
}

void Velocity3DPlot::wheelEvent(QWheelEvent *event) {
    double delta = event->angleDelta().y() > 0 ? 1.1 : 0.9;
    zoomFactor *= delta;
    zoomFactor = std::max(0.3, std::min(3.0, zoomFactor));
    update();
}

void Velocity3DPlot::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        isRotating = true;
        lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
    }
}

void Velocity3DPlot::mouseMoveEvent(QMouseEvent *event) {
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

void Velocity3DPlot::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        isRotating = false;
        setCursor(Qt::ArrowCursor);
    }
}

// ============================================================================
// VelocityModelWidget Implementation
// ============================================================================

VelocityModelWidget::VelocityModelWidget(QWidget *parent)
    : QWidget(parent), currentBoundary(nullptr), boundarySet(false)
{
    setupUI();
}

VelocityModelWidget::~VelocityModelWidget() {
    if (currentBoundary) {
        delete currentBoundary;
    }
}

void VelocityModelWidget::setupUI() {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    
    // Model type selection
    QGroupBox *typeGroup = new QGroupBox("Tipe Model Kecepatan", this);
    QVBoxLayout *typeLayout = new QVBoxLayout();
    
    modelTypeCombo = new QComboBox(this);
    modelTypeCombo->addItem("Homogen");
    modelTypeCombo->addItem("1D");
    modelTypeCombo->addItem("3D");
    connect(modelTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &VelocityModelWidget::onModelTypeChanged);
    typeLayout->addWidget(modelTypeCombo);
    
    typeGroup->setLayout(typeLayout);
    mainLayout->addWidget(typeGroup);
    
    // Model stack
    modelStack = new QStackedWidget(this);
    
    // ===== Homogeneous model widget =====
    homogeneousWidget = new QWidget();
    QFormLayout *homogLayout = new QFormLayout(homogeneousWidget);
    
    vpHomogeneous = new QLineEdit("6.0", this);
    vpHomogeneous->setValidator(new QDoubleValidator(0.1, 20.0, 3, this));
    homogLayout->addRow("Vp (km/s):", vpHomogeneous);
    
    QLabel *homogNote = new QLabel(
        "<b>Model Homogen:</b><br>"
        "Kecepatan konstan di seluruh domain.<br>"
        "Cocok untuk estimasi awal atau medium uniform.",
        this
    );
    homogNote->setWordWrap(true);
    homogNote->setStyleSheet("QLabel { background-color: #e8f5e9; padding: 8px; border-radius: 3px; }");
    homogLayout->addRow(homogNote);
    
    modelStack->addWidget(homogeneousWidget);
    
    // ===== 1D model widget =====
    model1DWidget = new QWidget();
    QVBoxLayout *model1DLayout = new QVBoxLayout(model1DWidget);
    
    load1DButton = new QPushButton("Load 1D Model File", this);
    load1DButton->setMinimumHeight(35);
    connect(load1DButton, &QPushButton::clicked, this, &VelocityModelWidget::onLoad1DModel);
    model1DLayout->addWidget(load1DButton);
    
    QLabel *format1DLabel = new QLabel(
        "<b>Format File 1D:</b> 2 kolom (Vp, MaxDepth)<br>"
        "Contoh: examples/velocity_model_1d.txt",
        this
    );
    format1DLabel->setWordWrap(true);
    model1DLayout->addWidget(format1DLabel);
    
    // Splitter for preview and plot
    QSplitter *splitter1D = new QSplitter(Qt::Horizontal, this);
    
    model1DPreview = new QTextEdit(this);
    model1DPreview->setReadOnly(true);
    model1DPreview->setMaximumHeight(250);
    model1DPreview->setPlaceholderText("File preview...");
    splitter1D->addWidget(model1DPreview);
    
    velocity1DPlot = new Velocity1DPlot(this);
    splitter1D->addWidget(velocity1DPlot);
    
    splitter1D->setStretchFactor(0, 1);
    splitter1D->setStretchFactor(1, 2);
    
    model1DLayout->addWidget(splitter1D);
    
    modelStack->addWidget(model1DWidget);
    
    // ===== 3D model widget =====
    model3DWidget = new QWidget();
    QVBoxLayout *model3DLayout = new QVBoxLayout(model3DWidget);
    
    load3DButton = new QPushButton("Load 3D Model File", this);
    load3DButton->setMinimumHeight(35);
    connect(load3DButton, &QPushButton::clicked, this, &VelocityModelWidget::onLoad3DModel);
    model3DLayout->addWidget(load3DButton);
    
    QLabel *format3DLabel = new QLabel(
        "<b>Format File 3D:</b> Lat, Lon, Depth, Vp<br>"
        "<b>Important:</b> Model grid harus sesuai dengan grid di 'Calculating Condition'<br>"
        "<b>Catatan:</b> Jika lokasi diluar model, gunakan nilai Vp terdekat<br>"
        "Contoh: examples/velocity_model_3d.txt",
        this
    );
    format3DLabel->setWordWrap(true);
    model3DLayout->addWidget(format3DLabel);
    
    // Location info label
    model3DLocationLabel = new QLabel("<i>Set boundary di 'Calculating Condition' terlebih dahulu</i>", this);
    model3DLocationLabel->setWordWrap(true);
    model3DLocationLabel->setStyleSheet("QLabel { background-color: #fff3cd; padding: 8px; border-radius: 3px; }");
    model3DLayout->addWidget(model3DLocationLabel);
    
    // Splitter for preview and plot
    QSplitter *splitter3D = new QSplitter(Qt::Horizontal, this);
    
    model3DPreview = new QTextEdit(this);
    model3DPreview->setReadOnly(true);
    model3DPreview->setMaximumHeight(250);
    model3DPreview->setPlaceholderText("File preview...");
    splitter3D->addWidget(model3DPreview);
    
    velocity3DPlot = new Velocity3DPlot(this);
    splitter3D->addWidget(velocity3DPlot);
    
    splitter3D->setStretchFactor(0, 1);
    splitter3D->setStretchFactor(1, 2);
    
    model3DLayout->addWidget(splitter3D);
    
    modelStack->addWidget(model3DWidget);
    
    mainLayout->addWidget(modelStack);
}

void VelocityModelWidget::onModelTypeChanged(int index) {
    // Clear all model data when switching to free memory
    model1DData.clear();
    model1DFilePath.clear();
    velocity1DPlot->clearData();
    
    model3DData.clear();
    model3DFilePath.clear();
    velocity3DPlot->clearData();
    
    modelStack->setCurrentIndex(index);
}

void VelocityModelWidget::onLoad1DModel() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Load 1D Velocity Model",
        "examples",
        "Text Files (*.txt);;All Files (*)"
    );
    
    if (fileName.isEmpty()) {
        return;
    }
    
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "Error", "Cannot open file: " + fileName);
        return;
    }
    
    QTextStream in(&file);
    QString content = in.readAll();
    file.close();
    
    // Parse data
    QStringList lines = content.split('\n', Qt::SkipEmptyParts);
    if (lines.size() < 2) {
        QMessageBox::warning(this, "Error", "File format tidak valid. Minimal 2 baris (header + data).");
        return;
    }
    
    // Check header
    QString header = lines[0].trimmed().toLower();
    if (!header.contains("vp") || !header.contains("maxdepth")) {
        QMessageBox::warning(this, "Error", "Header tidak sesuai format. Expected: Vp, MaxDepth");
        return;
    }
    
    // Parse layers
    model1DData.clear();
    for (int i = 1; i < lines.size(); ++i) {
        QStringList parts = lines[i].split(',');
        if (parts.size() >= 2) {
            VelocityLayer1D layer;
            layer.vp = parts[0].trimmed().toDouble();
            layer.maxDepth = parts[1].trimmed().toDouble();
            model1DData.append(layer);
        }
    }
    
    if (model1DData.isEmpty()) {
        QMessageBox::warning(this, "Error", "No valid data found in file");
        return;
    }
    
    model1DFilePath = fileName;
    model1DPreview->setPlainText(content);
    velocity1DPlot->setData(model1DData);
    
    QMessageBox::information(this, "Success", 
        QString("1D model loaded successfully!\nFile: %1\nLayers: %2")
            .arg(fileName).arg(model1DData.size()));
}

void VelocityModelWidget::onLoad3DModel() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Load 3D Velocity Model",
        "examples",
        "Text Files (*.txt);;All Files (*)"
    );
    
    if (fileName.isEmpty()) {
        return;
    }
    
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "Error", "Cannot open file: " + fileName);
        return;
    }
    
    QTextStream in(&file);
    QString previewContent;
    int lineCount = 0;
    int previewLines = 50;
    
    // Parse data
    QString header = in.readLine();
    previewContent += header + "\n";
    lineCount++;
    
    model3DData.clear();
    
    while (!in.atEnd()) {
        QString line = in.readLine();
        
        if (lineCount < previewLines) {
            previewContent += line + "\n";
        }
        
        QStringList parts = line.split(',');
        if (parts.size() >= 4) {
            VelocityPoint3D point;
            point.lat = parts[0].trimmed().toDouble();
            point.lon = parts[1].trimmed().toDouble();
            point.depth = parts[2].trimmed().toDouble();
            point.vp = parts[3].trimmed().toDouble();
            model3DData.append(point);
        }
        
        lineCount++;
    }
    
    file.close();
    
    if (lineCount > previewLines) {
        previewContent += QString("\n... (%1 more lines)\n").arg(lineCount - previewLines);
    }
    
    if (model3DData.isEmpty()) {
        QMessageBox::warning(this, "Error", "No valid data found in file");
        return;
    }
    
    model3DFilePath = fileName;
    model3DPreview->setPlainText(previewContent);
    velocity3DPlot->setData(model3DData);
    
    // Validate grid size if boundary is set
    if (boundarySet && currentBoundary) {
        // Calculate expected grid size
        double avgLat = (currentBoundary->yMin + currentBoundary->yMax) / 2.0;
        double latRad = avgLat * M_PI / 180.0;
        double xRangeKm = (currentBoundary->xMax - currentBoundary->xMin) * 111.320 * cos(latRad);
        double yRangeKm = (currentBoundary->yMax - currentBoundary->yMin) * 110.574;
        double zRangeKm = currentBoundary->depthMax - currentBoundary->depthMin;
        
        int expectedNX = static_cast<int>(xRangeKm / currentBoundary->gridSpacing) + 1;
        int expectedNY = static_cast<int>(yRangeKm / currentBoundary->gridSpacing) + 1;
        int expectedNZ = static_cast<int>(zRangeKm / currentBoundary->gridSpacing) + 1;
        long long expectedTotal = static_cast<long long>(expectedNX) * expectedNY * expectedNZ;
        
        if (model3DData.size() != expectedTotal) {
            QMessageBox::warning(this, "Grid Size Mismatch",
                QString("⚠️ 3D model size tidak sesuai dengan grid!\n\n"
                       "Expected: %1 points (%2×%3×%4)\n"
                       "Loaded: %5 points\n\n"
                       "Model harus sesuai dengan calculating condition grid.\n"
                       "Grid spacing: %6 km")
                    .arg(expectedTotal).arg(expectedNX).arg(expectedNY).arg(expectedNZ)
                    .arg(model3DData.size())
                    .arg(currentBoundary->gridSpacing, 0, 'f', 2));
        } else {
            QMessageBox::information(this, "Success", 
                QString("✓ 3D model loaded successfully!\n\n"
                       "File: %1\n"
                       "Points: %2 (%3×%4×%5)\n"
                       "Grid spacing: %6 km\n"
                       "Grid size matches calculating condition!")
                    .arg(fileName).arg(model3DData.size())
                    .arg(expectedNX).arg(expectedNY).arg(expectedNZ)
                    .arg(currentBoundary->gridSpacing, 0, 'f', 2));
        }
    } else {
        QMessageBox::information(this, "Success", 
            QString("3D model loaded successfully!\nFile: %1\nPoints: %2\n\n"
                   "Set boundary di 'Calculating Condition' untuk validasi grid.")
                .arg(fileName).arg(model3DData.size()));
    }
}

void VelocityModelWidget::setBoundary(const BoundaryData &boundary) {
    if (!currentBoundary) {
        currentBoundary = new BoundaryData();
    }
    *currentBoundary = boundary;
    boundarySet = true;
    
    // Update location label
    double avgLat = (boundary.yMin + boundary.yMax) / 2.0;
    double latRad = avgLat * M_PI / 180.0;
    double xRangeKm = (boundary.xMax - boundary.xMin) * 111.320 * cos(latRad);
    double yRangeKm = (boundary.yMax - boundary.yMin) * 110.574;
    double zRangeKm = boundary.depthMax - boundary.depthMin;
    
    int expectedNX = static_cast<int>(xRangeKm / boundary.gridSpacing) + 1;
    int expectedNY = static_cast<int>(yRangeKm / boundary.gridSpacing) + 1;
    int expectedNZ = static_cast<int>(zRangeKm / boundary.gridSpacing) + 1;
    
    model3DLocationLabel->setText(
        QString("<b>Calculating Condition Grid:</b><br>"
                "Location: [%1°, %2°] × [%3°, %4°] × [%5, %6] km<br>"
                "Grid: %7×%8×%9 points (spacing: %10 km)<br>"
                "<i>Model 3D harus memiliki %11 points total</i>")
            .arg(boundary.xMin, 0, 'f', 2).arg(boundary.xMax, 0, 'f', 2)
            .arg(boundary.yMin, 0, 'f', 2).arg(boundary.yMax, 0, 'f', 2)
            .arg(boundary.depthMin, 0, 'f', 1).arg(boundary.depthMax, 0, 'f', 1)
            .arg(expectedNX).arg(expectedNY).arg(expectedNZ)
            .arg(boundary.gridSpacing, 0, 'f', 2)
            .arg(static_cast<long long>(expectedNX) * expectedNY * expectedNZ)
    );
}

QString VelocityModelWidget::getModelType() const {
    return modelTypeCombo->currentText();
}

double VelocityModelWidget::getHomogeneousVp() const {
    return vpHomogeneous->text().toDouble();
}

QString VelocityModelWidget::get1DModelPath() const {
    return model1DFilePath;
}

QString VelocityModelWidget::get3DModelPath() const {
    return model3DFilePath;
}

QVector<VelocityLayer1D> VelocityModelWidget::get1DModelData() const {
    return model1DData;
}

QVector<VelocityPoint3D> VelocityModelWidget::get3DModelData() const {
    return model3DData;
}
