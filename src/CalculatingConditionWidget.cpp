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
    
    // Boundary Group
    QGroupBox *boundaryGroup = new QGroupBox("Boundary (km)", this);
    QFormLayout *boundaryLayout = new QFormLayout();
    
    QDoubleValidator *validator = new QDoubleValidator(this);
    
    xMinEdit = new QLineEdit("0.0", this);
    xMinEdit->setValidator(validator);
    boundaryLayout->addRow("X Min:", xMinEdit);
    
    xMaxEdit = new QLineEdit("100.0", this);
    xMaxEdit->setValidator(validator);
    boundaryLayout->addRow("X Max:", xMaxEdit);
    
    yMinEdit = new QLineEdit("0.0", this);
    yMinEdit->setValidator(validator);
    boundaryLayout->addRow("Y Min:", yMinEdit);
    
    yMaxEdit = new QLineEdit("100.0", this);
    yMaxEdit->setValidator(validator);
    boundaryLayout->addRow("Y Max:", yMaxEdit);
    
    zMinEdit = new QLineEdit("0.0", this);
    zMinEdit->setValidator(validator);
    boundaryLayout->addRow("Z Min:", zMinEdit);
    
    zMaxEdit = new QLineEdit("50.0", this);
    zMaxEdit->setValidator(validator);
    boundaryLayout->addRow("Z Max:", zMaxEdit);
    
    boundaryGroup->setLayout(boundaryLayout);
    mainLayout->addWidget(boundaryGroup);
    
    // Grid Group
    QGroupBox *gridGroup = new QGroupBox("Grid Spacing (km)", this);
    QFormLayout *gridLayout = new QFormLayout();
    
    dxEdit = new QLineEdit("1.0", this);
    dxEdit->setValidator(new QDoubleValidator(0.001, 1000.0, 3, this));
    gridLayout->addRow("dx (X-direction):", dxEdit);
    
    dyEdit = new QLineEdit("1.0", this);
    dyEdit->setValidator(new QDoubleValidator(0.001, 1000.0, 3, this));
    gridLayout->addRow("dy (Y-direction):", dyEdit);
    
    dzEdit = new QLineEdit("1.0", this);
    dzEdit->setValidator(new QDoubleValidator(0.001, 1000.0, 3, this));
    gridLayout->addRow("dz (Z-direction):", dzEdit);
    
    gridGroup->setLayout(gridLayout);
    mainLayout->addWidget(gridGroup);
    
    // Info Label
    QLabel *infoLabel = new QLabel(
        "<b>Note:</b><br>"
        "• Boundary mendefinisikan area perhitungan<br>"
        "• Grid spacing menentukan resolusi komputasi<br>"
        "• Grid lebih kecil = perhitungan lebih detail tapi lebih lama<br>"
        "• Klik 'Commit' untuk menerapkan kondisi ini",
        this
    );
    infoLabel->setWordWrap(true);
    infoLabel->setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }");
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
    connect(zMinEdit, &QLineEdit::textChanged, this, &CalculatingConditionWidget::validateInputs);
    connect(zMaxEdit, &QLineEdit::textChanged, this, &CalculatingConditionWidget::validateInputs);
}

void CalculatingConditionWidget::onCommitClicked() {
    BoundaryData boundary;
    
    // Parse values
    bool ok;
    boundary.xMin = xMinEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid X Min value"); return; }
    
    boundary.xMax = xMaxEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid X Max value"); return; }
    
    boundary.yMin = yMinEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid Y Min value"); return; }
    
    boundary.yMax = yMaxEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid Y Max value"); return; }
    
    boundary.zMin = zMinEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid Z Min value"); return; }
    
    boundary.zMax = zMaxEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid Z Max value"); return; }
    
    boundary.dx = dxEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid dx value"); return; }
    
    boundary.dy = dyEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid dy value"); return; }
    
    boundary.dz = dzEdit->text().toDouble(&ok);
    if (!ok) { QMessageBox::warning(this, "Error", "Invalid dz value"); return; }
    
    // Validate ranges
    if (boundary.xMin >= boundary.xMax) {
        QMessageBox::warning(this, "Error", "X Min harus lebih kecil dari X Max");
        return;
    }
    if (boundary.yMin >= boundary.yMax) {
        QMessageBox::warning(this, "Error", "Y Min harus lebih kecil dari Y Max");
        return;
    }
    if (boundary.zMin >= boundary.zMax) {
        QMessageBox::warning(this, "Error", "Z Min harus lebih kecil dari Z Max");
        return;
    }
    
    if (boundary.dx <= 0 || boundary.dy <= 0 || boundary.dz <= 0) {
        QMessageBox::warning(this, "Error", "Grid spacing harus lebih besar dari 0");
        return;
    }
    
    // Calculate grid size
    int nX = static_cast<int>((boundary.xMax - boundary.xMin) / boundary.dx) + 1;
    int nY = static_cast<int>((boundary.yMax - boundary.yMin) / boundary.dy) + 1;
    int nZ = static_cast<int>((boundary.zMax - boundary.zMin) / boundary.dz) + 1;
    
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
        QString("Calculating condition committed!\n"
               "Grid size: %1 × %2 × %3 = %4 points")
            .arg(nX).arg(nY).arg(nZ).arg(totalPoints));
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
    boundary.zMin = zMinEdit->text().toDouble();
    boundary.zMax = zMaxEdit->text().toDouble();
    boundary.dx = dxEdit->text().toDouble();
    boundary.dy = dyEdit->text().toDouble();
    boundary.dz = dzEdit->text().toDouble();
    return boundary;
}
