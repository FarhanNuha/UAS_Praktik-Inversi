#ifndef METHODWIDGET_H
#define METHODWIDGET_H

#include <QWidget>
#include <QComboBox>
#include <QStackedWidget>
#include <QLineEdit>
#include <QScrollArea>
#include <QCheckBox>
#include <QLabel>

// Forward declaration
struct BoundaryData;

class MethodWidget : public QWidget {
    Q_OBJECT

public:
    explicit MethodWidget(QWidget *parent = nullptr);
    ~MethodWidget();
    
    QString getApproach() const;
    QString getMethod() const;
    QString getGlobalSubMethod() const;

public slots:
    void updateGridSize(const BoundaryData &boundary);

private slots:
    void onApproachChanged(int index);
    void onLocalMethodChanged(int index);
    void onGlobalMethodChanged(int index);
    void onSAVariantChanged(int index);
    void onGAVariantChanged(int index);

private:
    void setupUI();
    void createLocalMethodWidgets();
    void createGlobalMethodWidgets();
    void createSimulatedAnnealingWidgets();
    void createGeneticAlgorithmWidgets();
    
    QComboBox *approachCombo;
    QStackedWidget *methodStack;
    
    // Local approach
    QWidget *localWidget;
    QComboBox *localMethodCombo;
    QStackedWidget *localParamStack;
    
    // Local method parameters
    QWidget *gaussNewtonParams;
    QWidget *steepestDescentParams;
    QWidget *levenbergMarquadtParams;
    
    // Global approach
    QWidget *globalWidget;
    QComboBox *globalMethodCombo;
    QStackedWidget *globalParamStack;
    
    // Global method parameters
    QWidget *gridSearchParams;
    QWidget *simulatedAnnealingParams;
    QWidget *geneticAlgorithmParams;
    
    // Grid Search widgets
    QLabel *gridSizeLabel;
    QCheckBox *monteCarloCheck;
    QLineEdit *sampleSizeEdit;
    
    // Simulated Annealing sub-widgets
    QComboBox *saVariantCombo;
    QStackedWidget *saVariantStack;
    QWidget *simpleSAParams;
    QWidget *metropolisSAParams;
    QWidget *cauchySAParams;
    
    // Genetic Algorithm sub-widgets
    QCheckBox *gaRealCodedCheck;
    QComboBox *gaVariantCombo;
    QStackedWidget *gaVariantStack;
    QWidget *standardGAParams;
    QWidget *steadyStateGAParams;
    QWidget *spea2Params;
};

#endif // METHODWIDGET_H
