"""Exact paper-identifier expressions for joins on Parquet inputs."""
import pyarrow as pa
import pyarrow.parquet as pq


def paper_id_sql(path, column, *, allow_float=False):
    dtype = pq.ParquetFile(path).schema_arrow.field(column).type
    name = '"' + column.replace('"', '""') + '"'
    if pa.types.is_floating(dtype):
        if not allow_float:
            raise ValueError('Citation IDs cannot be floating point.')
        return (f'CASE WHEN {name}>0 AND {name}<9007199254740992 '
                f'AND {name}=floor({name}) THEN CAST({name} AS UBIGINT) '
                "ELSE error('Unsafe floating-point paper identifier') END")
    if pa.types.is_integer(dtype):
        return f'CAST({name} AS UBIGINT)'
    return ("CAST(regexp_replace(trim(CAST(" + name + " AS VARCHAR)), "
            "'^(https?://openalex[.]org/)?W', '') AS UBIGINT)")
