#ifndef DATAINPUTWIDGET_H
#define DATAINPUTWIDGET_H

#include <QWidget>
#include <QPushButton>
#include <QTableWidget>
#include "MapViewer2D.h"

class DataInputWidget : public QWidget {
    Q_OBJECT

public:
    explicit DataInputWidget(QWidget *parent = nullptr);
    ~DataInputWidget();
    
    QVector<StationData> getStationData() const;

signals:
    void stationsLoaded(const QVector<StationData> &stations);
    void computationRequested();

private slots:
    void onLoadDataFile();
    void onAddRow();
    void onDeleteRow();
    void onComputeClicked();
    void onTableDataChanged();

private:
    void setupUI();
    void updateStationsFromTable();
    
    QPushButton *loadButton;
    QPushButton *addRowButton;
    QPushButton *deleteRowButton;
    QPushButton *computeButton;
    QTableWidget *stationTable;
    
    QVector<StationData> stations;
};

#endif // DATAINPUTWIDGET_H
