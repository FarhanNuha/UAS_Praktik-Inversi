#include "VelocityModelWidget.h"
#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QFileDialog>
#include <QMessageBox>
#include <QFile>
#include <QTextStream>
#include <QDoubleValidator>

VelocityModelWidget::VelocityModelWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

VelocityModelWidget::~VelocityModelWidget() {
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
    
    // Homogeneous model widget
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
    
    // 1D model widget
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
    
    model1DPreview = new QTextEdit(this);
    model1DPreview->setReadOnly(true);
    model1DPreview->setMaximumHeight(150);
    model1DPreview->setPlaceholderText("File preview akan tampil di sini...");
    model1DLayout->addWidget(model1DPreview);
    
    modelStack->addWidget(model1DWidget);
    
    // 3D model widget
    model3DWidget = new QWidget();
    QVBoxLayout *model3DLayout = new QVBoxLayout(model3DWidget);
    
    load3DButton = new QPushButton("Load 3D Model File", this);
    load3DButton->setMinimumHeight(35);
    connect(load3DButton, &QPushButton::clicked, this, &VelocityModelWidget::onLoad3DModel);
    model3DLayout->addWidget(load3DButton);
    
    QLabel *format3DLabel = new QLabel(
        "<b>Format File 3D:</b> 4 kolom (X, Y, Z, Vp)<br>"
        "Contoh: examples/velocity_model_3d.txt",
        this
    );
    format3DLabel->setWordWrap(true);
    model3DLayout->addWidget(format3DLabel);
    
    model3DPreview = new QTextEdit(this);
    model3DPreview->setReadOnly(true);
    model3DPreview->setMaximumHeight(150);
    model3DPreview->setPlaceholderText("File preview akan tampil di sini...");
    model3DLayout->addWidget(model3DPreview);
    
    modelStack->addWidget(model3DWidget);
    
    mainLayout->addWidget(modelStack);
    mainLayout->addStretch();
}

void VelocityModelWidget::onModelTypeChanged(int index) {
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
    
    // Validate format (simple check)
    QStringList lines = content.split('\n', Qt::SkipEmptyParts);
    if (lines.size() < 2) {
        QMessageBox::warning(this, "Error", "File format tidak valid. Minimal 2 baris (header + data).");
        return;
    }
    
    // Check header
    QString header = lines[0].trimmed().toLower();
    if (!header.contains("vp") || !header.contains("maxdepth")) {
        QMessageBox::warning(this, "Error", 
            "Header tidak sesuai format. Expected: Vp, MaxDepth");
        return;
    }
    
    model1DFilePath = fileName;
    model1DPreview->setPlainText(content);
    
    QMessageBox::information(this, "Success", 
        QString("1D model loaded successfully!\nFile: %1\nLines: %2")
            .arg(fileName).arg(lines.size()));
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
    QString content;
    int lineCount = 0;
    int previewLines = 100; // Only show first 100 lines in preview
    
    while (!in.atEnd() && lineCount < previewLines) {
        content += in.readLine() + "\n";
        lineCount++;
    }
    
    if (!in.atEnd()) {
        content += "\n... (file continues)\n";
    }
    
    file.close();
    
    // Validate format (simple check)
    QStringList lines = content.split('\n', Qt::SkipEmptyParts);
    if (lines.size() < 2) {
        QMessageBox::warning(this, "Error", "File format tidak valid. Minimal 2 baris (header + data).");
        return;
    }
    
    // Check header
    QString header = lines[0].trimmed().toLower();
    if (!header.contains("x") || !header.contains("y") || 
        !header.contains("z") || !header.contains("vp")) {
        QMessageBox::warning(this, "Error", 
            "Header tidak sesuai format. Expected: X, Y, Z, Vp");
        return;
    }
    
    model3DFilePath = fileName;
    model3DPreview->setPlainText(content);
    
    QMessageBox::information(this, "Success", 
        QString("3D model loaded successfully!\nFile: %1")
            .arg(fileName));
}

QString VelocityModelWidget::getModelType() const {
    return modelTypeCombo->currentText();
}
