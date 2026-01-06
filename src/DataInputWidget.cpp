#include "DataInputWidget.h"
#include "ComputationEngine.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QFileDialog>
#include <QMessageBox>
#include <QFile>
#include <QTextStream>
#include <QLabel>
#include <QRegularExpression>
#include <QProgressDialog>

DataInputWidget::DataInputWidget(QWidget *parent)
    : QWidget(parent), computationEngine(nullptr)
{
    setupUI();
    
    // Create computation engine
    computationEngine = new ComputationEngine(this);
    
    // Connect signals
    connect(computationEngine, &ComputationEngine::progressUpdated,
            this, &DataInputWidget::onComputationProgress);
    connect(computationEngine, &ComputationEngine::computationFinished,
            this, &DataInputWidget::onComputationFinished);
    connect(computationEngine, &ComputationEngine::computationError,
            this, &DataInputWidget::onComputationError);
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
    QString header = in.readLine();
    
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
            parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        } else {
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
    
    if (!boundaryData || !methodData || !velocityData) {
        QMessageBox::warning(this, "Warning", 
            "Setup tidak lengkap!\n\n"
            "Pastikan sudah:\n"
            "• Set Calculating Condition\n"
            "• Pilih Metode Inversi\n"
            "• Set Velocity Model");
        return;
    }
    
    QMessageBox::StandardButton reply = QMessageBox::question(
        this,
        "Konfirmasi Komputasi",
        QString("Mulai komputasi inversi hiposenter?\n\n"
               "Jumlah stasiun: %1\n"
               "Metode: %2\n"
               "Model kecepatan: %3\n\n"
               "Proses ini mungkin memakan waktu.")
            .arg(stations.size())
            .arg(methodData->methodName)
            .arg(velocityData->modelType),
        QMessageBox::Yes | QMessageBox::No
    );
    
    if (reply == QMessageBox::No) {
        return;
    }
    
    // Setup computation engine
    computationEngine->setBoundary(*boundaryData);
    computationEngine->setStations(stations);
    
    // Set velocity model
    if (velocityData->modelType == "Homogen") {
        computationEngine->setVelocityModel("Homogen", velocityData->homogeneousVp);
    }
    else if (velocityData->modelType == "1D") {
        computationEngine->setVelocityModel1D(velocityData->layers1D);
    }
    else if (velocityData->modelType == "3D") {
        computationEngine->setVelocityModel3D(velocityData->points3D);
    }
    
    // Create progress dialog
    progressDialog = new QProgressDialog("Memulai komputasi...", "Cancel", 0, 100, this);
    progressDialog->setWindowTitle("Computation Progress");
    progressDialog->setWindowModality(Qt::WindowModal);
    progressDialog->setMinimumDuration(0);
    progressDialog->show();
    
    connect(progressDialog, &QProgressDialog::canceled, this, [this]() {
        QMessageBox::information(this, "Cancelled", "Komputasi dibatalkan oleh user");
    });
    
    // Start computation based on method
    QString approach = methodData->approach;
    QString method = methodData->methodName;
    
    if (approach == "Pendekatan Lokal") {
        if (method == "Gauss Newton") {
            computationEngine->computeGaussNewton(1e-6, 100);
        }
        else if (method == "Steepest Descent") {
            computationEngine->computeSteepestDescent(1e-5, 200, 0.01);
        }
        else if (method == "Levenberg Marquardt") {
            computationEngine->computeLevenbergMarquardt(1e-6, 100, 0.01);
        }
    }
    else if (approach == "Pendekatan Global") {
        if (method == "Grid Search") {
            bool useMC = methodData->useMonteCarloSampling;
            int samples = methodData->monteCarloSamples;
            computationEngine->computeGridSearch(useMC, samples);
        }
        // Add other global methods as needed
    }
}

void DataInputWidget::onComputationProgress(int percent, const QString &message) {
    if (progressDialog) {
        progressDialog->setValue(percent);
        progressDialog->setLabelText(message);
    }
}

void DataInputWidget::onComputationFinished(const HypocenterResult &result) {
    if (progressDialog) {
        progressDialog->close();
        delete progressDialog;
        progressDialog = nullptr;
    }
    
    // Format result text
    QString resultText = QString(
        "╔═══════════════════════════════════════════════════════════════╗\n"
        "║       HASIL INVERSI LOKASI HIPOSENTER GEMPABUMI               ║\n"
        "╚═══════════════════════════════════════════════════════════════╝\n\n"
        "METODE INVERSI\n"
        "──────────────────────────────────────────────────────────────\n"
        "  Pendekatan    : %1\n"
        "  Metode        : %2\n"
        "  Iterasi       : %3\n"
        "  Konvergensi   : %4\n\n"
        "LOKASI HIPOSENTER\n"
        "──────────────────────────────────────────────────────────────\n"
        "  Longitude     : %5°\n"
        "  Latitude      : %6°\n"
        "  Depth         : %7 km\n"
        "  Origin Time   : %8 detik dari first arrival\n\n"
        "STATISTIK MISFIT\n"
        "──────────────────────────────────────────────────────────────\n"
        "  RMS Misfit    : %9 detik\n"
        "  Mean Residual : %10 detik\n"
        "  Std Residual  : %11 detik\n"
        "  Max Residual  : %12 detik\n"
        "  Min Residual  : %13 detik\n\n"
    ).arg(methodData->approach)
     .arg(methodData->methodName)
     .arg(result.iterations)
     .arg(result.converged ? "Ya ✓" : "Tidak ✗")
     .arg(result.x, 0, 'f', 6)
     .arg(result.y, 0, 'f', 6)
     .arg(result.z, 0, 'f', 3)
     .arg(result.originTime, 0, 'f', 3)
     .arg(result.rms, 0, 'f', 6)
     .arg(result.residuals.isEmpty() ? 0.0 : 
          std::accumulate(result.residuals.begin(), result.residuals.end(), 0.0) / result.residuals.size(), 0, 'f', 6)
     .arg(0.0, 0, 'f', 6) // Calculate std
     .arg(result.residuals.isEmpty() ? 0.0 : 
          *std::max_element(result.residuals.begin(), result.residuals.end()), 0, 'f', 6)
     .arg(result.residuals.isEmpty() ? 0.0 : 
          *std::min_element(result.residuals.begin(), result.residuals.end()), 0, 'f', 6);
    
    // Add per-station residuals
    resultText += "RESIDUAL PER STASIUN\n";
    resultText += "──────────────────────────────────────────────────────────────\n";
    for (int i = 0; i < stations.size() && i < result.residuals.size(); ++i) {
        resultText += QString("  %1: %2 detik\n")
            .arg(stations[i].name, -15)
            .arg(result.residuals[i], 8, 'f', 6);
    }
    resultText += "\n";
    resultText += "═══════════════════════════════════════════════════════════════\n";
    resultText += QString("Waktu komputasi: %1\n")
        .arg(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss"));
    resultText += "═══════════════════════════════════════════════════════════════\n";
    
    // Emit result to be displayed
    emit computationComplete(resultText, result);
    
    QMessageBox::information(this, "Computation Complete", 
        QString("Inversi hiposenter selesai!\n\n"
               "Lokasi: %1°, %2°\n"
               "Depth: %3 km\n"
               "RMS: %4 detik\n"
               "Iterasi: %5\n\n"
               "Lihat tab 'Hasil' untuk detail lengkap.")
            .arg(result.x, 0, 'f', 4)
            .arg(result.y, 0, 'f', 4)
            .arg(result.z, 0, 'f', 2)
            .arg(result.rms, 0, 'f', 6)
            .arg(result.iterations));
}

void DataInputWidget::onComputationError(const QString &error) {
    if (progressDialog) {
        progressDialog->close();
        delete progressDialog;
        progressDialog = nullptr;
    }
    
    QMessageBox::critical(this, "Computation Error", 
        QString("Terjadi error saat komputasi:\n\n%1").arg(error));
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
            continue;
        }
        
        station.arrivalTime = timeItem->text();
        
        stations.append(station);
    }
    
    computeButton->setEnabled(!stations.isEmpty());
    
    if (!stations.isEmpty()) {
        emit stationsLoaded(stations);
    }
}

void DataInputWidget::setBoundaryData(const BoundaryData &boundary) {
    if (!boundaryData) {
        boundaryData = new BoundaryData();
    }
    *boundaryData = boundary;
}

void DataInputWidget::setMethodData(const QString &approach, const QString &method, 
                                   bool useMC, int mcSamples) {
    if (!methodData) {
        methodData = new MethodData();
    }
    methodData->approach = approach;
    methodData->methodName = method;
    methodData->useMonteCarloSampling = useMC;
    methodData->monteCarloSamples = mcSamples;
}

void DataInputWidget::setVelocityData(const QString &type, double vp,
                                     const QVector<VelocityLayer1D> &layers,
                                     const QVector<VelocityPoint3D> &points) {
    if (!velocityData) {
        velocityData = new VelocityData();
    }
    velocityData->modelType = type;
    velocityData->homogeneousVp = vp;
    velocityData->layers1D = layers;
    velocityData->points3D = points;
}

QVector<StationData> DataInputWidget::getStationData() const {
    return stations;
}
