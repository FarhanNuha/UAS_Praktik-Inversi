#include "DataInputWidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QFileDialog>
#include <QMessageBox>
#include <QFile>
#include <QTextStream>
#include <QLabel>
#include <QRegularExpression>

DataInputWidget::DataInputWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

DataInputWidget::~DataInputWidget() {
}

void DataInputWidget::setupUI() {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(10);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    
    // Separator selection
    QHBoxLayout *separatorLayout = new QHBoxLayout();
    separatorLayout->addWidget(new QLabel("File Separator:", this));
    
    separatorCombo = new QComboBox(this);
    separatorCombo->addItem("Comma (,)");
    separatorCombo->addItem("Space/Tab");
    separatorLayout->addWidget(separatorCombo);
    separatorLayout->addStretch();
    
    mainLayout->addLayout(separatorLayout);
    
    // Load file button
    loadButton = new QPushButton("Load Station Data File", this);
    loadButton->setMinimumHeight(35);
    connect(loadButton, &QPushButton::clicked, this, &DataInputWidget::onLoadDataFile);
    mainLayout->addWidget(loadButton);
    
    // Format info
    QLabel *formatLabel = new QLabel(
        "<b>Format:</b> Stasiun, Latitude, Longitude, Arrival Time UTC (hh:mm:ss)<br>"
        "<i>Contoh: examples/station_data.txt</i>",
        this
    );
    formatLabel->setWordWrap(true);
    mainLayout->addWidget(formatLabel);
    
    // Table widget
    stationTable = new QTableWidget(0, 4, this);
    stationTable->setHorizontalHeaderLabels({"Stasiun", "Latitude", "Longitude", "Arrival Time (UTC)"});
    stationTable->horizontalHeader()->setStretchLastSection(true);
    stationTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    stationTable->setAlternatingRowColors(true);
    
    connect(stationTable, &QTableWidget::cellChanged, this, &DataInputWidget::onTableDataChanged);
    
    mainLayout->addWidget(stationTable);
    
    // Button row
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    
    addRowButton = new QPushButton("Add Row", this);
    connect(addRowButton, &QPushButton::clicked, this, &DataInputWidget::onAddRow);
    buttonLayout->addWidget(addRowButton);
    
    deleteRowButton = new QPushButton("Delete Row", this);
    connect(deleteRowButton, &QPushButton::clicked, this, &DataInputWidget::onDeleteRow);
    buttonLayout->addWidget(deleteRowButton);
    
    buttonLayout->addStretch();
    
    mainLayout->addLayout(buttonLayout);
    
    // Compute button
    computeButton = new QPushButton("HITUNG", this);
    computeButton->setMinimumHeight(45);
    computeButton->setStyleSheet(
        "QPushButton { "
        "background-color: #2196F3; "
        "color: white; "
        "font-weight: bold; "
        "font-size: 14px; "
        "border-radius: 5px; "
        "}"
        "QPushButton:hover { background-color: #1976D2; }"
        "QPushButton:pressed { background-color: #0D47A1; }"
        "QPushButton:disabled { background-color: #BDBDBD; }"
    );
    connect(computeButton, &QPushButton::clicked, this, &DataInputWidget::onComputeClicked);
    mainLayout->addWidget(computeButton);
    
    // Initially disable compute button
    computeButton->setEnabled(false);
}

void DataInputWidget::onLoadDataFile() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Load Station Data",
        "examples",
        "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
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
    QString header = in.readLine(); // Skip header
    
    // Check header format
    QString headerLower = header.toLower();
    if (!headerLower.contains("stasiun") || !headerLower.contains("latitude") || 
        !headerLower.contains("longitude") || !headerLower.contains("arrival")) {
        QMessageBox::warning(this, "Error", 
            "Header tidak sesuai format.\nExpected: Stasiun, Latitude, Longitude, Arrival Time UTC");
        file.close();
        return;
    }
    
    // Determine separator
    bool useSpace = (separatorCombo->currentIndex() == 1);
    
    // Clear existing data
    stationTable->setRowCount(0);
    stations.clear();
    
    int row = 0;
    int skippedLines = 0;
    
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty()) continue;
        
        QStringList parts;
        if (useSpace) {
            // Split by whitespace (space or tab), remove empty entries
            parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        } else {
            // Split by comma
            parts = line.split(',');
        }
        
        if (parts.size() < 4) {
            skippedLines++;
            continue;
        }
        
        stationTable->insertRow(row);
        
        for (int col = 0; col < 4 && col < parts.size(); ++col) {
            QTableWidgetItem *item = new QTableWidgetItem(parts[col].trimmed());
            stationTable->setItem(row, col, item);
        }
        
        row++;
    }
    
    file.close();
    
    updateStationsFromTable();
    
    QString message = QString("Loaded %1 stations from file:\n%2").arg(row).arg(fileName);
    if (skippedLines > 0) {
        message += QString("\n\n%1 invalid lines were skipped").arg(skippedLines);
    }
    
    QMessageBox::information(this, "Success", message);
}

void DataInputWidget::onAddRow() {
    int row = stationTable->rowCount();
    stationTable->insertRow(row);
    
    // Set default values
    stationTable->setItem(row, 0, new QTableWidgetItem(QString("STA%1").arg(row + 1)));
    stationTable->setItem(row, 1, new QTableWidgetItem("0.0"));
    stationTable->setItem(row, 2, new QTableWidgetItem("0.0"));
    stationTable->setItem(row, 3, new QTableWidgetItem("00:00:00"));
}

void DataInputWidget::onDeleteRow() {
    int currentRow = stationTable->currentRow();
    if (currentRow >= 0) {
        stationTable->removeRow(currentRow);
        updateStationsFromTable();
    } else {
        QMessageBox::information(this, "Info", "Please select a row to delete");
    }
}

void DataInputWidget::onComputeClicked() {
    if (stations.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Tidak ada data stasiun untuk diproses!");
        return;
    }
    
    QMessageBox::StandardButton reply = QMessageBox::question(
        this,
        "Konfirmasi Komputasi",
        QString("Mulai komputasi dengan %1 stasiun?\n\n"
               "Pastikan:\n"
               "• Calculating Condition sudah di-commit\n"
               "• Metode sudah dipilih\n"
               "• Velocity model sudah di-set\n\n"
               "Proses ini mungkin memakan waktu lama.")
            .arg(stations.size()),
        QMessageBox::Yes | QMessageBox::No
    );
    
    if (reply == QMessageBox::Yes) {
        emit computationRequested();
        QMessageBox::information(this, "Computation", 
            "Komputasi dimulai!\n(Placeholder - implementasi perhitungan akan ditambahkan)");
    }
}

void DataInputWidget::onTableDataChanged() {
    updateStationsFromTable();
}

void DataInputWidget::updateStationsFromTable() {
    stations.clear();
    
    for (int row = 0; row < stationTable->rowCount(); ++row) {
        StationData station;
        
        QTableWidgetItem *nameItem = stationTable->item(row, 0);
        QTableWidgetItem *latItem = stationTable->item(row, 1);
        QTableWidgetItem *lonItem = stationTable->item(row, 2);
        QTableWidgetItem *timeItem = stationTable->item(row, 3);
        
        if (!nameItem || !latItem || !lonItem || !timeItem) {
            continue;
        }
        
        station.name = nameItem->text();
        
        bool ok1, ok2;
        station.latitude = latItem->text().toDouble(&ok1);
        station.longitude = lonItem->text().toDouble(&ok2);
        
        if (!ok1 || !ok2) {
            continue; // Skip invalid entries
        }
        
        station.arrivalTime = timeItem->text();
        
        stations.append(station);
    }
    
    // Enable compute button if we have valid stations
    computeButton->setEnabled(!stations.isEmpty());
    
    // Emit signal to update map viewers
    if (!stations.isEmpty()) {
        emit stationsLoaded(stations);
    }
}

QVector<StationData> DataInputWidget::getStationData() const {
    return stations;
}
