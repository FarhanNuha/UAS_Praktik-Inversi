#include "ResultWidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QMessageBox>
#include <QDateTime>

ResultWidget::ResultWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

ResultWidget::~ResultWidget() {
}

void ResultWidget::setupUI() {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(10);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    
    // Result text area
    resultText = new QTextEdit(this);
    resultText->setReadOnly(true);
    resultText->setPlaceholderText("Hasil perhitungan akan ditampilkan di sini...\n\n"
                                   "Informasi yang akan ditampilkan:\n"
                                   "• Parameter metode yang digunakan\n"
                                   "• Iterasi dan konvergensi\n"
                                   "• Lokasi hasil inversi (X, Y, Z)\n"
                                   "• Origin time\n"
                                   "• Misfit dan residual\n"
                                   "• Statistik hasil");
    
    QFont font("Courier New", 10);
    resultText->setFont(font);
    
    mainLayout->addWidget(resultText);
    
    // Button layout
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    
    clearButton = new QPushButton("Clear", this);
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
}
