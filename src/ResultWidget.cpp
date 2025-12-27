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
    hasData = !iterations.isEmpty();
    update();
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

void Result2DPlot::setResult(double x, double y, const QVector<QPointF> &contour) {
    resultX = x;
    resultY = y;
    contourData = contour;
    hasData = true;
    update();
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
    painter.drawText(QRect(0, 5, width(), 20), Qt::AlignCenter, "Result 2D - XY Plane");
    
    int centerX = width() / 2;
    int centerY = height() / 2;
    int scale = 100;
    
    // Draw axes
    painter.setPen(QPen(Qt::black, 2));
    painter.drawLine(centerX - scale, centerY, centerX + scale, centerY);
    painter.drawLine(centerX, centerY - scale, centerX, centerY + scale);
    
    // Draw contour (placeholder)
    painter.setPen(QPen(QColor(100, 100, 255), 1));
    for (int i = 1; i <= 5; ++i) {
        int radius = i * 15;
        painter.drawEllipse(QPointF(centerX, centerY), radius, radius);
    }
    
    // Draw result point
    painter.setPen(QPen(Qt::red, 3));
    painter.setBrush(Qt::red);
    painter.drawEllipse(QPointF(centerX, centerY), 8, 8);
    
    painter.setPen(Qt::red);
    painter.drawText(centerX + 15, centerY - 10, QString("Result: (%1, %2)").arg(resultX, 0, 'f', 2).arg(resultY, 0, 'f', 2));
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

void Result3DPlot::setResult(double x, double y, double z) {
    resultX = x;
    resultY = y;
    resultZ = z;
    hasData = true;
    update();
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
    painter.drawText(QRect(0, 5, width(), 20), Qt::AlignCenter, "Result 3D");
    
    int centerX = width() / 2;
    int centerY = height() / 2 + 10;
    double scale = 80 * zoomFactor;
    
    // Draw 3D axes
    painter.setPen(QPen(Qt::red, 2));
    painter.drawLine(project3D(0, 0, 0), project3D(1, 0, 0));
    painter.drawText(project3D(1.2, 0, 0) + QPointF(5, 0), "X");
    
    painter.setPen(QPen(Qt::green, 2));
    painter.drawLine(project3D(0, 0, 0), project3D(0, 1, 0));
    painter.drawText(project3D(0, 1.2, 0) + QPointF(5, 0), "Y");
    
    painter.setPen(QPen(Qt::blue, 2));
    painter.drawLine(project3D(0, 0, 0), project3D(0, 0, 1));
    painter.drawText(project3D(0, 0, 1.2) + QPointF(-15, 0), "Z");
    
    // Draw result point
    QPointF resultPos = project3D(resultX, resultY, resultZ);
    painter.setPen(QPen(Qt::red, 3));
    painter.setBrush(Qt::red);
    painter.drawEllipse(resultPos, 8, 8);
    
    painter.setPen(Qt::darkRed);
    painter.drawText(resultPos + QPointF(12, 0), QString("(%1, %2, %3)")
        .arg(resultX, 0, 'f', 2).arg(resultY, 0, 'f', 2).arg(resultZ, 0, 'f', 2));
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

void ResultWidget::set2DResult(double x, double y, const QVector<QPointF> &contour) {
    result2DPlot->setResult(x, y, contour);
}

void ResultWidget::set3DResult(double x, double y, double z) {
    result3DPlot->setResult(x, y, z);
}
