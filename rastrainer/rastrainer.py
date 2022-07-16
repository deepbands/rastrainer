# -*- coding: utf-8 -*-
"""
/***************************************************************************
 rastrainer
                                 A QGIS plugin
 rastrainer is a QGIS plugin to training remote sensing semantic segmentation model based on PaddlePaddle.


 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2021-12-11
        git sha              : $Format:%H$
        copyright            : (C) 2021 by deepbands
        email                : geoyee@yeah.net
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .rastrainer_dialog import rastrainerDialog
import os.path

# add
import os.path as osp
from qgis.core import QgsMapLayerProxyModel
from .utils import (
    MODELS, Model, QTrainDaraset, QEvalDaraset, Raster
)


class rastrainer:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'rastrainer_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&rastrainer')

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

        # add
        self.batch_size_list = [str(2 ** i) for i in range(10)]
        self.log_list = [str(10 * i) for i in range(1, 10)]

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('rastrainer', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/rastrainer/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'rastrainer '),
            callback=self.run,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.first_start = True


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&rastrainer'),
                action)
            self.iface.removeToolBarIcon(action)


    def change_classes(self):
        self.num_class = int(self.dlg.edtClasses.text())


    def select_model(self):
        self.model_name = self.dlg.cbxModel.currentText()
        
    
    def select_params_file(self):
        param_file = self.dlg.mQfwPretrained.filePath()
        self.param_file = param_file if param_file != "" else None


    def run(self):
        """Run method that performs all the real work"""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = rastrainerDialog()
            self.num_class = 2
            self.model_name = MODELS[0]
            self.param_file = None
        # setting
        self.dlg.mcbxRaster.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.dlg.mcbxMask.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.dlg.mcbxShp.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self.dlg.mQfwPretrained.setFilter("*.pdparams")
        self.dlg.mQfwPretrained.fileChanged.connect(self.select_params_file)
        self.dlg.cbxModel.addItems(MODELS)
        self.dlg.cbxModel.currentTextChanged.connect(self.select_model)
        self.dlg.cbxBatch.addItems(self.batch_size_list)
        self.dlg.cbxLog.addItems(self.log_list)
        self.dlg.edtClasses.textChanged.connect(self.change_classes)

        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            self.num_class = int(self.dlg.edtClasses.text())
            self.modeler = Model(
                self.model_name,
                self.num_class,
                self.param_file
            )
            # TODO: TEST
            image_path = "testdata/test.tif"
            label_path = "testdata/lab2.tif"
            train_datas = QTrainDaraset(image_path, label_path, self.num_class)
            val_datas = None

            args = {
                "learning_rate": float(self.dlg.edtLearning.text()),
                "epochs": int(self.dlg.edtEpoch.text()),
                "batch_size": int(self.dlg.cbxBatch.currentText()),
                "train_dataset":train_datas,
                "val_dataset": val_datas,
                "save_dir": self.dlg.mQfwOutput.filePath(),
                "save_number": int(self.dlg.edtEval.text()),
                "log_iters": int(self.dlg.cbxLog.currentText())
            }

            self.modeler.train(args)