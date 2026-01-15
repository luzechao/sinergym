#!/usr/bin/env python3
"""
Script para verificar que las acciones se están aplicando correctamente en las observaciones.

Reglas de comparación:
1. Las acciones en la fila i corresponden a las observaciones en la fila i+1 (por el reset)
2. Para flow_rates:
   - Si acción = 0.0, observación debe ser 0.0 (válvula cerrada)
   - Si acción = 1.0, observación debe ser > 0 (válvula abierta, flujo máximo)
3. Para water_temperature: debe ser igual (con tolerancia de punto flotante)
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_csv_file(file_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Lee un archivo CSV y retorna los headers y las filas como diccionarios.

    Args:
        file_path: Ruta al archivo CSV

    Returns:
        Tupla con (headers, lista de diccionarios con los datos)
    """
    headers: List[str] = []
    rows: List[Dict[str, str]] = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"El archivo CSV '{file_path}' no tiene headers válidos.")
        headers = list(reader.fieldnames)

        for row in reader:
            rows.append(row)

    return headers, rows


def compare_actions_with_observations(
    actions_path: str, observations_path: str, tolerance: float = 1e-6
) -> Dict[str, List[Tuple[int, str, str, str]]]:
    """
    Compara las acciones con las observaciones.

    Args:
        actions_path: Ruta al archivo CSV de acciones
        observations_path: Ruta al archivo CSV de observaciones
        tolerance: Tolerancia para comparación de valores flotantes

    Returns:
        Diccionario con errores encontrados: {columna: [(fila_action, valor_action, valor_obs, mensaje), ...]}
    """
    # Leer ambos archivos
    action_headers, action_rows = read_csv_file(actions_path)
    obs_headers, obs_rows = read_csv_file(observations_path)

    # Encontrar columnas comunes
    common_columns = set(action_headers) & set(obs_headers)

    if not common_columns:
        print("ERROR: No hay columnas comunes entre los dos archivos CSV.")
        return {}

    print(f"Columnas comunes encontradas: {sorted(common_columns)}")
    print(f"Acciones tiene {len(action_rows)} filas de datos")
    print(f"Observaciones tiene {len(obs_rows)} filas de datos")
    print()

    # Errores encontrados por columna
    errors: Dict[str, List[Tuple[int, str, str, str]]] = defaultdict(list)

    # Comparar: actions[i] con observations[i+1]
    # actions[0] -> observations[1]
    # actions[1] -> observations[2]
    # etc.

    max_comparisons = min(len(action_rows), len(obs_rows) - 1)

    if max_comparisons == 0:
        print("ADVERTENCIA: No hay suficientes filas para comparar.")
        print("Observaciones necesita al menos 2 filas (header + 1 fila de datos)")
        return {}

    print(f"Comparando {max_comparisons} filas...")
    print(f"(Acciones fila i -> Observaciones fila i+1)")
    print()

    # Para cada fila de acciones
    for i in range(max_comparisons):
        action_row = action_rows[i]
        obs_row = obs_rows[i + 1]  # Observaciones con offset de 1

        # Comparar cada columna común
        for column in common_columns:
            action_value_str = action_row.get(column, '').strip()
            obs_value_str = obs_row.get(column, '').strip()

            # Determinar tipo de comparación según el nombre de la columna
            is_flow_rate_column = 'flow_rate' in column.lower()
            is_water_temp_column = column == 'water_temperature'

            try:
                if is_flow_rate_column:
                    # Para flow_rates: lógica especial
                    action_value = float(action_value_str)
                    obs_value = float(obs_value_str)

                    # Caso 1: Acción = 0.0 -> Observación debe ser 0.0
                    if action_value == 0.0:
                        if obs_value != 0.0:
                            errors[column].append(
                                (
                                    i
                                    + 2,  # +2 porque: +1 por índice base 0->1, +1 por header
                                    f"{action_value:.6f}",
                                    f"{obs_value:.6f}",
                                    f"Acción es 0.0 (cerrado) pero observación es {obs_value:.6f} (debería ser 0.0)",
                                )
                            )

                    # Caso 2: Acción = 1.0 -> Observación debe ser > 0
                    elif action_value == 1.0:
                        if obs_value <= 0.0:
                            errors[column].append(
                                (
                                    i + 2,
                                    f"{action_value:.6f}",
                                    f"{obs_value:.6f}",
                                    f"Acción es 1.0 (abierto) pero observación es {obs_value:.6f} (debería ser > 0)",
                                )
                            )

                    # Caso 3: Acción tiene otro valor (no debería pasar, pero lo verificamos)
                    else:
                        # Si acción no es 0.0 ni 1.0, tratamos como error de formato
                        errors[column].append(
                            (
                                i + 2,
                                f"{action_value:.6f}",
                                f"{obs_value:.6f}",
                                f"Acción tiene valor inesperado: {action_value:.6f} (debería ser 0.0 o 1.0)",
                            )
                        )

                elif is_water_temp_column:
                    # Para water_temperature: comparación con tolerancia
                    action_value = float(action_value_str)
                    obs_value = float(obs_value_str)

                    if abs(action_value - obs_value) > tolerance:
                        errors[column].append(
                            (
                                i + 2,
                                f"{action_value:.6f}",
                                f"{obs_value:.6f}",
                                f"Diferencia: {abs(action_value - obs_value):.6f} (tolerancia: {tolerance})",
                            )
                        )

                else:
                    # Para otras columnas: comparación exacta (por si acaso hay más columnas comunes)
                    if action_value_str != obs_value_str:
                        errors[column].append(
                            (
                                i + 2,
                                action_value_str,
                                obs_value_str,
                                "Valores no coinciden",
                            )
                        )

            except (ValueError, TypeError) as e:
                # Error al convertir a float
                errors[column].append(
                    (
                        i + 2,
                        action_value_str,
                        obs_value_str,
                        f"Error al convertir valores: {str(e)}",
                    )
                )

    return dict(errors)


def main():
    """Función principal del script."""
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(
            "Uso: python verificar_acciones.py <actions_csv> <observations_csv> [tolerance]"
        )
        print()
        print("Compara las acciones con las observaciones:")
        print("  - Acciones fila i -> Observaciones fila i+1")
        print("  - Para flow_rates: acción 0.0 -> obs 0.0, acción 1.0 -> obs > 0")
        print("  - Para water_temperature: comparación con tolerancia (default: 1e-6)")
        sys.exit(1)

    actions_path = sys.argv[1]
    observations_path = sys.argv[2]
    tolerance = float(sys.argv[3]) if len(sys.argv) == 4 else 1e-6

    # Verificar que los archivos existen
    if not Path(actions_path).exists():
        print(f"ERROR: El archivo '{actions_path}' no existe.")
        sys.exit(1)

    if not Path(observations_path).exists():
        print(f"ERROR: El archivo '{observations_path}' no existe.")
        sys.exit(1)

    print("=" * 80)
    print("VERIFICACIÓN DE ACCIONES vs OBSERVACIONES")
    print("=" * 80)
    print(f"Acciones: {actions_path}")
    print(f"Observaciones: {observations_path}")
    print(f"Tolerancia: {tolerance}")
    print("=" * 80)
    print()

    # Realizar la comparación
    errors = compare_actions_with_observations(
        actions_path, observations_path, tolerance
    )

    # Mostrar resultados
    print("=" * 80)
    print("RESULTADOS")
    print("=" * 80)

    if not errors:
        print("\n✓ Todas las acciones se están aplicando correctamente.")
        print("  No se encontraron errores en ninguna columna.")
    else:
        total_errors = sum(len(error_list) for error_list in errors.values())
        print(
            f"\n✗ Se encontraron {total_errors} error(es) en {len(errors)} columna(s):"
        )
        print()

        for column in sorted(errors.keys()):
            error_list = errors[column]
            print(f"  Columna: {column}")
            print(f"    Total de errores: {len(error_list)}")

            # Mostrar primeros 10 errores
            for i, (row_num, action_val, obs_val, message) in enumerate(
                error_list[:10]
            ):
                print(f"    Error {i+1} en fila {row_num} de acciones:")
                print(f"      Acción: {action_val}")
                print(f"      Observación: {obs_val}")
                print(f"      {message}")

            if len(error_list) > 10:
                print(f"    ... y {len(error_list) - 10} error(es) más")
            print()

    print("=" * 80)

    # Exit code: 0 si todo está bien, 1 si hay errores
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
