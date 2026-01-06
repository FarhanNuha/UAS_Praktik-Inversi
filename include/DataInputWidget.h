#ifndef DATAINPUTWIDGET_H
#define DATAINPUTWIDGET_H

#include <QWidget>
#include <QPushButton>
#include <QTableWidget>
#include <QComboBox>
#include <QProgressDialog>
#include "SharedTypes.h"

// Forward declaration
class ComputationEngine;

class DataInputWidget : public QWidget {
    Q_OBJECT

public:
    explicit DataInputWidget(QWidget *parent = nullptr);
    ~DataInputWidget();
    
    QVector<StationData> getStationData() const;
    
    // Setup functions for computation
    void setBoundaryData(const BoundaryData &boundary);
    void setMethodData(const QString &approach, const QString &method,
                      bool useMC = false, int mcSamples = 1000);
    void setVelocityData(const QString &type, double vp = 6.0,
                        const QVector<VelocityLayer1D> &layers = QVector<VelocityLayer1D>(),
                        const QVector<VelocityPoint3D> &points = QVector<VelocityPoint3D>());

signals:
    void stationsLoaded(const QVector<StationData> &stations);
    void computationRequested();
    void computationComplete(const QString &resultText, const HypocenterResult &result);

private slots:
    void onLoadDataFile();
    void onAddRow();
    void onDeleteRow();
    void onComputeClicked();
    void onTableDataChanged();
    void onComputationProgress(int percent, const QString &message);
    void onComputationFinished(const HypocenterResult &result);
    void onComputationError(const QString &error);

private:
    void setupUI();
    void updateStationsFromTable();
    
    QPushButton *loadButton;
    QPushButton *addRowButton;
    QPushButton *deleteRowButton;
    QPushButton *computeButton;
    QTableWidget *stationTable;
    QComboBox *separatorCombo;
    
    QVector<StationData> stations;
    
    // Computation
    ComputationEngine *computationEngine;
    QProgressDialog *progressDialog;
    
    // Data storage for computation
    BoundaryData *boundaryData = nullptr;
    MethodData *methodData = nullptr;
    VelocityData *velocityData = nullptr;
};

#endif // DATAINPUTWIDGET_H
