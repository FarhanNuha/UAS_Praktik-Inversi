#include "CalculatingConditionWidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QMessageBox>
#include <QDoubleValidator>

CalculatingConditionWidget::CalculatingConditionWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

CalculatingConditionWidget::~CalculatingConditionWidget() {
}

void CalculatingConditionWidget::setupUI() {
    QScrollArea *scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    
    QWidget *contentWidget = new QWidget();
    QVBoxLayout *mainLayout = new QVBoxLayout(contentWidget);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    
    // Horizontal Boundary Group (X, Y)
    QGroupBox *horizontalGroup = new QGroupBox("Horizontal Boundary (Longitude, Latitude)", this);
    QFormLayout *horizontalLayout = new QFormLayout();
    
    QDoubleValidator *coordValidator = new QDoubleValidator(-180.0, 180.0, 6, this);
    
    xMinEdit = new QLineEdit("95.0", this);
    xMinEdit->setValidator(coordValidator);
    horizontalLayout->addRow("Longitude Min (°):", xMinEdit);
    
    xMaxEdit = new QLineEdit("141.0", this);
    xMaxEdit->setValidator(coordValidator);
    horizontalLayout->addRow("Longitude Max (°):", xMaxEdit);
    
    QDoubleValidator *latValidator = new QDoubleValidator(-90.0, 90.0, 6, this);
    
    yMinEdit = new QLineEdit("-11.0", this);
    yMinEdit->setValidator(latValidator);
    horizontalLayout->addRow("Latitude Min (°):", yMinEdit);
    
    yMaxEdit = new QLineEdit("6.0", this);
    yMaxEdit->setValidator(latValidator);
    horizontalLayout->addRow("Latitude Max (°):", yMaxEdit);
    
    horizontalGroup->setLayout(horizontalLayout);
    mainLayout->addWidget(horizontalGroup);
    
    // Depth Group
    QGroupBox *depthGroup = new QGroupBox("Depth Range (km)", this);
    QFormLayout *depthLayout = new QFormLayout();
    
    QDoubleValidator *depthValidator = new QDoubleValidator(0.0, 1000.0, 3, this);
    
    depthMinEdit = new QLineEdit("0.0", this);
    depthMinEdit->setValidator(depthValidator);
    depthLayout->addRow("Depth Min:", depthMinEdit);
    
    depthMaxEdit = new QLineEdit("50.0", this);
    depthMaxEdit->setValidator(depthValidator);
    depthLayout->addRow("Depth Max:", depthMaxEdit);
    
    depthGroup->setLayout(depthLayout);
    mainLayout->addWidget(depthGroup);
    
    // Grid Spacing Group
    QGroupBox *gridGroup = new QGroupBox("Grid Spacing (km)", this);
    QFormLayout *gridLayout = new QFormLayout();
    
    gridSpacingEdit = new QLineEdit("1.0", this);
    gridSpacingEdit->setValidator(new QDoubleValidator(0.001, 1000.0, 3, this));
    gridLayout->addRow("Grid Spacing (dx = dy = dz):", gridSpacingEdit);
    
    QLabel *gridNote = new QLabel(
        "<i>Grid spacing yang sama untuk arah X, Y, dan Z<br>"
        "Spacing lebih kecil = resolusi lebih tinggi</i>",
        this
    );
    gridNote->setWordWrap(true);
    gridNote->setStyleSheet("QLabel { color: #666; padding: 5px; }");
    gridLayout->addRow(gridNote);
    
    gridGroup->setLayout(gridLayout);
    mainLayout->addWidget(gridGroup);
    
    // Info Label
    QLabel *infoLabel = new QLabel(
        "<b>Catatan:</b><br>"
        "• Boundary mendefinisikan area perhitungan<br>"
        "• Grid spacing menentukan resolusi komputasi<br>"
        "• Grid lebih kecil = perhitungan lebih detail tapi lebih lama<br>"
        "• Depth: kedalaman dari permukaan (0 km) ke bawah<br>"
        "• Klik 'Commit' untuk menerapkan kondisi ini",
        this
    );
    infoLabel->setWordWrap(true);
    infoLabel->setStyleSheet("QLabel { background-color: #e3f2fd; padding: 10px; border-radius: 5px; }");
    mainLayout->addWidget(infoLabel);
    
    // Commit Button
    commitButton = new QPushButton("Commit", this);
    commitButton->setMinimumHeight(40);
    commitButton->setStyleSheet(
        "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; border-radius: 5px; }"
        "QPushButton:hover { background-color: #45a049; }"
        "QPushButton:pressed { background-color: #3d8b40; }"
    );
    connect(commitButton, &QPushButton::clicked, this, &CalculatingConditionWidget::onCommitClicked);
    mainLayout->addWidget(commitButton);
    
    mainLayout->addStretch();
    
    scrollArea->setWidget(contentWidget);
    
    QVBoxLayout *outerLayout = new QVBoxLayout(this);
    outerLayout->setContentsMargins(0, 0, 0, 0);
    outerLayout->addWidget(scrollArea);
    
    // Connect validation
    connect(xMinEdit, &QLineEdit::textChanged, this, &CalculatingConditionWidget::validateInputs);
    connect(xMaxEdit, &QLineEdit::textChanged, this, &CalculatingConditionWidget::validateInputs);
    connect(yMinEdit, &QLineEdit::textChanged, this, &CalculatingConditionWidget::validateInputs);
    connect(yMaxEdit, &QLineEdit::textChanged, this, &CalculatingConditionWidget::validateInputs);
    connect(depthMinEdit, &QLineEdit::textChanged, this, &CalculatingConditionWidget::validateInputs);
    connect(depthMaxEdit, &QLineEdit::textChanged, this, &CalculatingConditionWidget::validateInputs);
}

void CalculatingConditionWidget::onCommitClicked() {
    BoundaryData boundary;
    
    // Parse values
    bool ok;
    boundary.xMin = xMinEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid Longitude Min value"); return; }
    
    boundary.xMax = xMaxEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid Longitude Max value"); return; }
    
    boundary.yMin = yMinEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid Latitude Min value"); return; }
    
    boundary.yMax = yMaxEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid Latitude Max value"); return; }
    
    boundary.depthMin = depthMinEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid Depth Min value"); return; }
    
    boundary.depthMax = depthMaxEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid Depth Max value"); return; }
    
    boundary.gridSpacing = gridSpacingEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid Grid Spacing value"); return; }
    
    // Validate ranges
    if (boundary.xMin >= boundary.xMax) {
        QMessageBox::warning(this, "Error", "Longitude Min harus lebih kecil dari Longitude Max");
        return;
    }
    if (boundary.yMin >= boundary.yMax) {
        QMessageBox::warning(this, "Error", "Latitude Min harus lebih kecil dari Latitude Max");
        return;
    }
    if (boundary.depthMin >= boundary.depthMax) {
        QMessageBox::warning(this, "Error", "Depth Min harus lebih kecil dari Depth Max");
        return;
    }
    
    if (boundary.gridSpacing <= 0) {
        QMessageBox::warning(this, "Error", "Grid spacing harus lebih besar dari 0");
        return;
    }
    
    // Calculate grid size
    int nX = static_cast<int>((boundary.xMax - boundary.xMin) / boundary.gridSpacing) + 1;
    int nY = static_cast<int>((boundary.yMax - boundary.yMin) / boundary.gridSpacing) + 1;
    int nZ = static_cast<int>((boundary.depthMax - boundary.depthMin) / boundary.gridSpacing) + 1;
    
    long long totalPoints = static_cast<long long>(nX) * nY * nZ;
    
    if (totalPoints > 10000000) { // 10 million points
        QMessageBox::StandardButton reply = QMessageBox::question(
            this, 
            "Warning", 
            QString("Grid size sangat besar: %1 × %2 × %3 = %4 points\n"
                   "Ini dapat memakan waktu komputasi yang sangat lama.\n"
                   "Lanjutkan?")
                .arg(nX).arg(nY).arg(nZ).arg(totalPoints),
            QMessageBox::Yes | QMessageBox::No
        );
        
        if (reply == QMessageBox::No) {
            return;
        }
    }
    
    emit conditionCommitted(boundary);
    
    QMessageBox::information(this, "Success", 
        QString("Calculating condition committed!\n\n"
               "Horizontal Grid: %1 × %2\n"
               "Depth Layers: %3\n"
               "Total Points: %4\n"
               "Grid Spacing: %5 km")
            .arg(nX).arg(nY).arg(nZ).arg(totalPoints)
            .arg(boundary.gridSpacing, 0, 'f', 3));
}

void CalculatingConditionWidget::validateInputs() {
    // Could add real-time validation feedback here
}

BoundaryData CalculatingConditionWidget::getBoundaryData() const {
    BoundaryData boundary;
    boundary.xMin = xMinEdit->text().toDouble();
    boundary.xMax = xMaxEdit->text().toDouble();
    boundary.yMin = yMinEdit->text().toDouble();
    boundary.yMax = yMaxEdit->text().toDouble();
    boundary.depthMin = depthMinEdit->text().toDouble();
    boundary.depthMax = depthMaxEdit->text().toDouble();
    boundary.gridSpacing = gridSpacingEdit->text().toDouble();
    return boundary;
}
